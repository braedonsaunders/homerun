"""Entropy-weighted maker strategy for crypto binary markets.

Detects opportunities directly from ``crypto_update`` worker payloads by
scoring binary entropy combined with spread quality and cancellation rate.
Does not depend on other crypto strategies for signal generation.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any

from models import Market, Opportunity
from services.data_events import DataEvent, EventType
from services.quality_filter import QualityFilterOverrides
from services.strategies._firehose import (
    GateResult,
    MURMUR,
    WHISPER,
    emit_emit_nowait,
    emit_evaluation_nowait,
)
from services.strategies.base import BaseStrategy, DecisionCheck, ExitDecision, StrategyDecision, _trader_size_limits
from services.strategy_helpers.crypto_strategy_utils import (
    build_binary_crypto_market,
    default_max_market_data_age_ms,
    default_max_oracle_age_ms,
    default_min_seconds_left_for_entry,
    fee_aware_min_edge_pct,
    history_cancel_peak,
    normalize_ratio,
    normalize_signed_ratio,
    parse_datetime_utc,
    pick_oracle_source,
    polymarket_taker_fee_pct,
)
from services.trader_orchestrator.strategies.sizing import compute_position_size
from utils.converters import clamp, safe_float, to_confidence, to_float
from utils.signal_helpers import selected_probability, signal_payload

logger = logging.getLogger(__name__)


def _probability_entropy(prob_yes: float) -> float:
    p = min(0.999999, max(0.000001, float(prob_yes)))
    q = 1.0 - p
    entropy = -(p * math.log(p, 2.0)) - (q * math.log(q, 2.0))
    return max(0.0, min(1.0, entropy))


class CryptoEntropyMakerStrategy(BaseStrategy):
    strategy_type = "crypto_entropy_maker"
    name = "Crypto Entropy Maker"
    description = "Canonical crypto microstructure strategy using entropy, cancel recovery, and orderflow quality."
    mispricing_type = "within_market"
    source_key = "crypto"
    market_categories = ["crypto"]
    subscriptions = [EventType.CRYPTO_UPDATE]
    supports_entry_take_profit_exit = True
    default_open_order_timeout_seconds = 20.0

    quality_filter_overrides = QualityFilterOverrides(
        min_roi=0.6,
        max_resolution_months=0.1,
    )

    default_config = {
        "min_edge_percent": 1.0,
        "min_confidence": 0.40,
        "min_entropy": 0.82,
        "min_spread_pct": 0.006,
        "max_spread_pct": 0.065,
        "max_spread_widening_bps": 22.0,
        "max_cancel_rate_30s": 0.75,
        "min_prior_peak_cancel_rate": 0.80,
        "min_cancel_drop": 0.14,
        "min_orderflow_alignment": 0.05,
        "min_recent_move_zscore": 1.25,
        "min_liquidity_usd": 1000.0,
        "max_entry_price": 0.92,
        "max_markets_per_event": 24,
        "min_order_size_usd": 2.0,
        "sizing_policy": "adaptive",
        "take_profit_pct": 6.5,
        "stop_loss_pct": 4.0,
        "max_hold_minutes": 16.0,
        # Safety gates — leave at None to use the timeframe-aware defaults in
        # crypto_strategy_utils. The maker path doesn't pay a taker fee on
        # rest, but if a maker order ages past the resolution boundary it
        # ends up filling as a stale taker, so we still apply the same gates.
        "max_oracle_age_ms": None,
        "max_market_data_age_ms": None,
        "min_seconds_left_for_entry": None,
        "min_fee_clearance_x": 1.5,
        "prefer_oracle_source": "binance_direct",
        # Exit any open position when this close to the resolution boundary
        # so we don't ride the Chainlink heartbeat window with a held leg.
        # 30s is comfortably past the longest open-order timeout below
        # while still leaving a meaningful runway after entry.
        "force_exit_seconds_before_resolution": 30.0,
        # Add the maker's open-order timeout to the entry gate so a fresh
        # rest order can't be placed close enough to resolution that it
        # ages out *into* the resolution window. Effective gate becomes
        # ``default_min_seconds_left_for_entry(tf) + open_order_timeout``.
        "maker_rest_includes_timeout": True,
    }

    def __init__(self) -> None:
        super().__init__()
        self.min_profit = 0.0

    @staticmethod
    def _row_market(row: dict[str, Any]) -> Market | None:
        return build_binary_crypto_market(row)

    def _score_market(self, row: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any] | None:
        gates: list[GateResult] = []

        def _emit_reject(verbosity: str = MURMUR) -> None:
            emit_evaluation_nowait(
                strategy_slug="crypto_entropy_maker",
                market=row,
                gates=gates,
                outcome="rejected",
                verbosity=verbosity,
            )

        up_price = safe_float(row.get("up_price"), None)
        down_price = safe_float(row.get("down_price"), None)
        prices_present = up_price is not None and down_price is not None
        gates.append(GateResult(
            "prices_present", "Up/down prices present", prices_present,
            detail=f"up={up_price} down={down_price}",
        ))
        if not prices_present:
            _emit_reject(WHISPER)
            return None
        prices_in_range = 0.0 < up_price < 1.0 and 0.0 < down_price < 1.0
        gates.append(GateResult(
            "prices_in_range", "Prices in (0, 1)", prices_in_range,
            detail=f"up={up_price:.4f} down={down_price:.4f}",
        ))
        if not prices_in_range:
            _emit_reject(WHISPER)
            return None

        # ---- Resolution-boundary + freshness gates (mirrors spike strategy)
        timeframe_value = row.get("timeframe")
        end_date = parse_datetime_utc(row.get("end_time"))
        seconds_left = float("inf")
        if end_date is not None:
            seconds_left = max(0.0, (end_date - datetime.now(timezone.utc)).total_seconds())
        configured_min_secs = cfg.get("min_seconds_left_for_entry")
        min_seconds_left = (
            float(configured_min_secs)
            if configured_min_secs is not None
            else default_min_seconds_left_for_entry(timeframe_value)
        )
        # Maker orders may rest unfilled for up to ``open_order_timeout``
        # before the orchestrator cancels them. Inflate the entry gate by
        # that window so a rest order that times out doesn't age out into
        # the resolution boundary.
        if bool(cfg.get("maker_rest_includes_timeout", True)):
            timeout_seconds = float(self.default_open_order_timeout_seconds or 0.0)
            min_seconds_left += max(0.0, timeout_seconds)
        seconds_left_ok = seconds_left >= min_seconds_left
        gates.append(GateResult(
            "min_seconds_left", "Seconds left for entry", seconds_left_ok,
            score=float(seconds_left) if seconds_left != float("inf") else None,
            detail=f"left={seconds_left:.1f}s min={min_seconds_left:.1f}s",
        ))
        if not seconds_left_ok:
            _emit_reject(MURMUR)
            return None

        configured_max_md_age = cfg.get("max_market_data_age_ms")
        max_market_data_age_ms = (
            float(configured_max_md_age)
            if configured_max_md_age is not None
            else default_max_market_data_age_ms(timeframe_value)
        )
        market_data_age_ms = safe_float(row.get("market_data_age_ms"), None)
        md_age_ok = market_data_age_ms is None or market_data_age_ms <= max_market_data_age_ms
        gates.append(GateResult(
            "market_data_age", "Worker snapshot fresh", md_age_ok,
            score=market_data_age_ms,
            detail=f"age_ms={market_data_age_ms} max={max_market_data_age_ms:.0f}",
        ))
        if not md_age_ok:
            _emit_reject(MURMUR)
            return None

        # Binary entropy
        total = up_price + down_price
        prob_yes = up_price / total if total > 0 else 0.5
        entropy = _probability_entropy(prob_yes)

        min_entropy = max(0.0, min(1.0, to_float(cfg.get("min_entropy", 0.82), 0.82)))
        entropy_ok = entropy >= min_entropy
        gates.append(GateResult(
            "min_entropy", "Min binary entropy", entropy_ok,
            score=float(entropy),
            detail=f"H={entropy:.4f} min={min_entropy:.3f}",
        ))
        if not entropy_ok:
            _emit_reject(MURMUR)
            return None

        # Spread
        spread = safe_float(row.get("spread"), None)
        min_spread_pct = max(0.0, min(1.0, to_float(cfg.get("min_spread_pct", 0.006), 0.006)))
        max_spread_pct = max(min_spread_pct, min(1.0, to_float(cfg.get("max_spread_pct", 0.065), 0.065)))
        spread_ok = spread is not None and min_spread_pct <= spread <= max_spread_pct
        gates.append(GateResult(
            "spread_window", "Spread within window", spread_ok,
            score=spread,
            detail=f"spread={spread} range=[{min_spread_pct:.4f},{max_spread_pct:.4f}]",
        ))
        if not spread_ok:
            _emit_reject(MURMUR)
            return None

        # Cancel rate
        cancel_rate_30s = normalize_ratio(row.get("cancel_rate_30s") or row.get("maker_cancel_rate_30s"))
        max_cancel_rate_30s = max(0.0, min(1.0, to_float(cfg.get("max_cancel_rate_30s", 0.75), 0.75)))
        cancel_rate_ok = cancel_rate_30s is None or cancel_rate_30s <= max_cancel_rate_30s
        gates.append(GateResult(
            "cancel_rate", "Cancel rate <= max", cancel_rate_ok,
            score=cancel_rate_30s,
            detail=f"rate={cancel_rate_30s} max={max_cancel_rate_30s:.3f}",
        ))
        if not cancel_rate_ok:
            _emit_reject(MURMUR)
            return None

        spread_widening_bps = safe_float(row.get("spread_widening_bps"), None)
        max_spread_widening_bps = max(0.0, to_float(cfg.get("max_spread_widening_bps", 22.0), 22.0))
        spread_widening_ok = spread_widening_bps is None or spread_widening_bps <= max_spread_widening_bps
        gates.append(GateResult(
            "spread_widening", "Spread widening within cap", spread_widening_ok,
            score=spread_widening_bps,
            detail=f"widening_bps={spread_widening_bps} max={max_spread_widening_bps:.1f}",
        ))
        if not spread_widening_ok:
            _emit_reject(MURMUR)
            return None

        prior_peak_cancel_rate = normalize_ratio(row.get("cancel_peak_2m"))
        if prior_peak_cancel_rate is None:
            prior_peak_cancel_rate = history_cancel_peak(row.get("history_tail"))
        cancel_recovery = None
        min_prior_peak_cancel_rate = max(
            0.0,
            min(1.0, to_float(cfg.get("min_prior_peak_cancel_rate", 0.80), 0.80)),
        )
        min_cancel_drop = max(0.0, min(1.0, to_float(cfg.get("min_cancel_drop", 0.14), 0.14)))
        if cancel_rate_30s is not None and prior_peak_cancel_rate is not None:
            recovered = prior_peak_cancel_rate - cancel_rate_30s
            if prior_peak_cancel_rate >= min_prior_peak_cancel_rate and recovered >= min_cancel_drop:
                cancel_recovery = recovered

        flow_imbalance = normalize_signed_ratio(row.get("flow_imbalance") or row.get("orderflow_imbalance"))
        if flow_imbalance is not None:
            flow_imbalance = abs(flow_imbalance)
        min_orderflow_alignment = max(
            0.0,
            min(1.0, to_float(cfg.get("min_orderflow_alignment", 0.05), 0.05)),
        )
        if flow_imbalance is not None and flow_imbalance < min_orderflow_alignment:
            flow_imbalance = None

        recent_move_zscore = safe_float(row.get("recent_move_zscore"), None)
        min_recent_move_zscore = max(0.0, to_float(cfg.get("min_recent_move_zscore", 1.25), 1.25))
        if recent_move_zscore is not None and abs(recent_move_zscore) < min_recent_move_zscore:
            recent_move_zscore = None

        # Liquidity
        liquidity = max(0.0, float(safe_float(row.get("liquidity"), 0.0) or 0.0))
        min_liquidity_usd = max(0.0, to_float(cfg.get("min_liquidity_usd", 1000.0), 1000.0))
        liquidity_ok = liquidity >= min_liquidity_usd
        gates.append(GateResult(
            "min_liquidity", "Min liquidity", liquidity_ok,
            score=float(liquidity),
            detail=f"liq=${liquidity:.0f} min=${min_liquidity_usd:.0f}",
        ))
        if not liquidity_ok:
            _emit_reject(MURMUR)
            return None

        # ---- Oracle: prefer the freshest source (binance_direct sub-second
        # WS by default) and refuse if every source is past the staleness
        # cap. Falls through to price-skew direction if oracle is unavailable.
        configured_max_oracle = cfg.get("max_oracle_age_ms")
        max_oracle_age_ms = (
            float(configured_max_oracle)
            if configured_max_oracle is not None
            else default_max_oracle_age_ms(timeframe_value)
        )
        prefer_source = str(cfg.get("prefer_oracle_source") or "binance_direct").strip() or None
        oracle_pick = pick_oracle_source(
            row,
            prefer=prefer_source,
            max_age_ms=max_oracle_age_ms,
        )
        if oracle_pick is not None:
            oracle_price = float(oracle_pick["price"])
            oracle_age_ms = float(oracle_pick["age_ms"])
            oracle_source_used = str(oracle_pick["source"])
        else:
            fallback_price = safe_float(row.get("oracle_price"), None)
            fallback_age_seconds = safe_float(row.get("oracle_age_seconds"), None)
            if (
                fallback_price is not None
                and fallback_age_seconds is not None
                and fallback_age_seconds * 1000.0 <= max_oracle_age_ms
            ):
                oracle_price = float(fallback_price)
                oracle_age_ms = float(fallback_age_seconds) * 1000.0
                oracle_source_used = str(row.get("oracle_source") or "primary")
            else:
                oracle_price = None
                oracle_age_ms = None
                oracle_source_used = None

        price_to_beat = safe_float(row.get("price_to_beat"), None)

        if oracle_price is not None and price_to_beat is not None and price_to_beat > 0:
            diff_pct = ((oracle_price - price_to_beat) / price_to_beat) * 100.0
            if diff_pct > 0.0:
                direction = "buy_yes"
                outcome = "YES"
            else:
                direction = "buy_no"
                outcome = "NO"
        else:
            # Determine direction from price skew
            if up_price < 0.5:
                direction = "buy_yes"
                outcome = "YES"
            else:
                direction = "buy_no"
                outcome = "NO"
            diff_pct = abs(0.5 - up_price) * 100.0

        # Entry price
        if direction == "buy_yes":
            entry_price = float(up_price)
        else:
            entry_price = float(down_price)

        max_entry_price = clamp(to_float(cfg.get("max_entry_price", 0.92), 0.92), 0.05, 0.99)
        entry_price_ok = 0.0 < entry_price < 1.0 and entry_price <= max_entry_price
        gates.append(GateResult(
            "entry_price", "Entry price within bounds", entry_price_ok,
            score=float(entry_price),
            detail=f"price={entry_price:.4f} max={max_entry_price:.4f} outcome={outcome}",
        ))
        if not entry_price_ok:
            _emit_reject(MURMUR)
            return None

        # Entropy-weighted edge
        entropy_multiplier = 0.55 + (0.80 * entropy)
        edge = abs(diff_pct) * entropy_multiplier
        if cancel_recovery is not None:
            edge += cancel_recovery * 5.0
        if flow_imbalance is not None:
            edge += flow_imbalance * 1.5
        if recent_move_zscore is not None:
            edge += min(3.0, abs(recent_move_zscore)) * 0.4

        min_edge_percent = max(0.0, to_float(cfg.get("min_edge_percent", 1.0), 1.0))
        edge_ok = edge >= min_edge_percent
        gates.append(GateResult(
            "min_edge", "Min edge", edge_ok,
            score=float(edge),
            detail=f"edge={edge:.3f}% min={min_edge_percent:.3f}%",
        ))
        if not edge_ok:
            _emit_reject(MURMUR)
            return None

        # ---- Fee-aware gate: maker rests pay zero, but if the maker order
        # ages out and fills as taker (or the price moves and we cross the
        # spread on a follow-up), the realised cost is the taker curve. A
        # 1.5× clearance (vs spike's 2×) reflects the maker's discount.
        min_fee_clearance_x = max(1.0, to_float(cfg.get("min_fee_clearance_x", 1.5), 1.5))
        min_edge_required = fee_aware_min_edge_pct(entry_price, min_fee_clearance_x)
        fee_clearance_ok = edge >= min_edge_required
        gates.append(GateResult(
            "fee_clearance", "Edge clears fee threshold", fee_clearance_ok,
            score=float(edge),
            detail=f"edge={edge:.3f}% min={min_edge_required:.3f}% (× {min_fee_clearance_x:.1f})",
        ))
        if not fee_clearance_ok:
            _emit_reject(MURMUR)
            return None
        taker_fee_pct_value = polymarket_taker_fee_pct(entry_price) * 100.0

        # Confidence
        confidence = clamp(
            0.50
            + clamp(entropy - 0.5, 0, 0.20)
            + clamp(abs(diff_pct or 0) / 10.0, 0, 0.15)
            + clamp((1 - (spread or 0) / 0.065) * 0.08, 0, 0.08),
            0.40,
            0.92,
        )
        if cancel_recovery is not None:
            confidence = clamp(confidence + clamp(cancel_recovery * 0.12, 0.0, 0.08), 0.40, 0.92)
        if flow_imbalance is not None:
            confidence = clamp(confidence + clamp(flow_imbalance * 0.08, 0.0, 0.05), 0.40, 0.92)
        if recent_move_zscore is not None:
            confidence = clamp(
                confidence + clamp(abs(recent_move_zscore) / 6.0, 0.0, 0.04),
                0.40,
                0.92,
            )

        min_confidence = to_confidence(cfg.get("min_confidence", 0.40), 0.40)
        confidence_ok = confidence >= min_confidence
        gates.append(GateResult(
            "min_confidence", "Min confidence", confidence_ok,
            score=float(confidence),
            detail=f"conf={confidence:.3f} min={min_confidence:.3f}",
        ))
        if not confidence_ok:
            _emit_reject(MURMUR)
            return None

        # Score
        score = (
            (edge * 0.45)
            + (confidence * 30.0)
            + (min(1.0, entropy) * 10.0)
            + (min(1.0, liquidity / 20_000.0) * 4.0)
        )

        gates.append(GateResult(
            "score_market_all", "All scoring gates passed", True,
            score=float(edge),
            detail=f"edge={edge:.3f}% conf={confidence:.3f} entropy={entropy:.3f}",
        ))

        return {
            "_firehose_gates": list(gates),
            "direction": direction,
            "outcome": outcome,
            "entry_price": entry_price,
            "up_price": float(up_price),
            "down_price": float(down_price),
            "oracle_price": float(oracle_price) if oracle_price is not None else None,
            "oracle_source_used": oracle_source_used,
            "oracle_age_ms": float(oracle_age_ms) if oracle_age_ms is not None else None,
            "price_to_beat": float(price_to_beat) if price_to_beat is not None else None,
            "diff_pct": float(diff_pct),
            "entropy": float(entropy),
            "entropy_multiplier": float(entropy_multiplier),
            "edge": float(edge),
            "taker_fee_pct": float(taker_fee_pct_value),
            "confidence": float(confidence),
            "spread": float(spread),
            "spread_widening_bps": float(spread_widening_bps) if spread_widening_bps is not None else None,
            "cancel_rate_30s": float(cancel_rate_30s) if cancel_rate_30s is not None else None,
            "prior_peak_cancel_rate": (
                float(prior_peak_cancel_rate) if prior_peak_cancel_rate is not None else None
            ),
            "cancel_recovery": float(cancel_recovery) if cancel_recovery is not None else None,
            "flow_imbalance": float(flow_imbalance) if flow_imbalance is not None else None,
            "recent_move_zscore": float(recent_move_zscore) if recent_move_zscore is not None else None,
            "liquidity": float(liquidity),
            "seconds_left": float(seconds_left) if seconds_left != float("inf") else None,
            "timeframe": str(timeframe_value or "") or None,
            "score": float(score),
        }

    def _build_opportunity(self, row: dict[str, Any], signal: dict[str, Any]) -> Opportunity | None:
        typed_market = self._row_market(row)
        if typed_market is None:
            return None

        strategy_cfg = dict(getattr(self, "config", None) or {})
        direction = str(signal["direction"])
        outcome = str(signal["outcome"])
        entry_price = float(signal["entry_price"])
        token_ids = list(typed_market.clob_token_ids or [])
        token_idx = 0 if direction == "buy_yes" else 1
        token_id = token_ids[token_idx] if len(token_ids) > token_idx else None

        entropy_context = {
            "strategy_origin": "crypto_worker",
            "direction": direction,
            "entry_price": entry_price,
            "up_price": signal["up_price"],
            "down_price": signal["down_price"],
            "oracle_price": signal["oracle_price"],
            "oracle_source_used": signal.get("oracle_source_used"),
            "oracle_age_ms": signal.get("oracle_age_ms"),
            "price_to_beat": signal["price_to_beat"],
            "diff_pct": signal["diff_pct"],
            "entropy": signal["entropy"],
            "entropy_multiplier": signal["entropy_multiplier"],
            "edge": signal["edge"],
            "taker_fee_pct": signal.get("taker_fee_pct"),
            "confidence": signal["confidence"],
            "spread": signal["spread"],
            "cancel_rate_30s": signal["cancel_rate_30s"],
            "liquidity": signal["liquidity"],
            "seconds_left": signal.get("seconds_left"),
            "timeframe": signal.get("timeframe"),
            "market_data_age_ms": safe_float(row.get("market_data_age_ms"), None),
            "fetched_at": row.get("fetched_at"),
            "end_time": row.get("end_time"),
        }

        positions = [
            {
                "action": "BUY",
                "outcome": outcome,
                "price": entry_price,
                "token_id": token_id,
                "price_policy": "maker_limit",
                "time_in_force": "GTC",
                "_entropy": entropy_context,
            }
        ]

        opp = self.create_opportunity(
            title=f"Entropy Maker: {outcome} @ {entry_price:.3f} (H={signal['entropy']:.3f})",
            description=(
                f"entropy {signal['entropy']:.3f}, edge {signal['edge']:.2f}%, "
                f"spread {signal['spread']:.4f}"
            ),
            total_cost=entry_price,
            expected_payout=entry_price + (signal["edge"] / 100.0),
            markets=[typed_market],
            positions=positions,
            is_guaranteed=False,
            skip_fee_model=True,
            custom_roi_percent=signal["edge"],
            custom_risk_score=clamp(0.55 - (signal["entropy"] * 0.15), 0.15, 0.85),
            confidence=signal["confidence"],
            min_liquidity_hard=max(100.0, to_float(strategy_cfg.get("min_liquidity_usd", 1000.0), 1000.0)),
            min_position_size=max(1.0, to_float(strategy_cfg.get("min_order_size_usd", 2.0), 2.0)),
        )
        if opp is None:
            return None

        opp.risk_factors = [
            f"Binary entropy={signal['entropy']:.3f}",
            f"Edge={signal['edge']:.2f}% (entropy-weighted)",
            f"Spread={signal['spread']:.4f}",
        ]
        opp.strategy_context = {
            "source_key": "crypto",
            "strategy_slug": self.strategy_type,
            **entropy_context,
        }
        return opp

    def _rejection_reason(self, row: dict[str, Any], cfg: dict[str, Any]) -> str:
        up_price = safe_float(row.get("up_price"), None)
        down_price = safe_float(row.get("down_price"), None)
        if up_price is None or down_price is None:
            return "missing_prices"
        if not (0.0 < up_price < 1.0 and 0.0 < down_price < 1.0):
            return "invalid_prices"

        total = up_price + down_price
        prob_yes = up_price / total if total > 0 else 0.5
        entropy = _probability_entropy(prob_yes)
        min_entropy = max(0.0, min(1.0, to_float(cfg.get("min_entropy", 0.82), 0.82)))
        if entropy < min_entropy:
            return "low_entropy"

        spread = safe_float(row.get("spread"), None)
        min_spread_pct = max(0.0, min(1.0, to_float(cfg.get("min_spread_pct", 0.006), 0.006)))
        max_spread_pct = max(min_spread_pct, min(1.0, to_float(cfg.get("max_spread_pct", 0.065), 0.065)))
        if spread is None:
            return "missing_spread"
        if spread < min_spread_pct:
            return "spread_too_narrow"
        if spread > max_spread_pct:
            return "spread_too_wide"

        cancel_rate_30s = normalize_ratio(row.get("cancel_rate_30s") or row.get("maker_cancel_rate_30s"))
        max_cancel_rate_30s = max(0.0, min(1.0, to_float(cfg.get("max_cancel_rate_30s", 0.75), 0.75)))
        if cancel_rate_30s is not None and cancel_rate_30s > max_cancel_rate_30s:
            return "cancel_rate_too_high"

        spread_widening_bps = safe_float(row.get("spread_widening_bps"), None)
        max_spread_widening_bps = max(0.0, to_float(cfg.get("max_spread_widening_bps", 22.0), 22.0))
        if spread_widening_bps is not None and spread_widening_bps > max_spread_widening_bps:
            return "spread_widening"

        liquidity = max(0.0, float(safe_float(row.get("liquidity"), 0.0) or 0.0))
        min_liquidity_usd = max(0.0, to_float(cfg.get("min_liquidity_usd", 1000.0), 1000.0))
        if liquidity < min_liquidity_usd:
            return "low_liquidity"

        oracle_price = safe_float(row.get("oracle_price"), None)
        price_to_beat = safe_float(row.get("price_to_beat"), None)
        if oracle_price is not None and price_to_beat is not None and price_to_beat > 0:
            diff_pct = ((oracle_price - price_to_beat) / price_to_beat) * 100.0
            direction = "buy_yes" if diff_pct > 0.0 else "buy_no"
        else:
            direction = "buy_yes" if up_price < 0.5 else "buy_no"
            diff_pct = abs(0.5 - up_price) * 100.0

        entry_price = float(up_price if direction == "buy_yes" else down_price)
        max_entry_price = clamp(to_float(cfg.get("max_entry_price", 0.92), 0.92), 0.05, 0.99)
        if entry_price <= 0.0 or entry_price >= 1.0:
            return "invalid_entry_price"
        if entry_price > max_entry_price:
            return "entry_price_too_high"

        entropy_multiplier = 0.55 + (0.80 * entropy)
        edge = abs(diff_pct) * entropy_multiplier
        cancel_recovery = None
        prior_peak_cancel_rate = normalize_ratio(row.get("cancel_peak_2m"))
        if prior_peak_cancel_rate is None:
            prior_peak_cancel_rate = history_cancel_peak(row.get("history_tail"))
        min_prior_peak_cancel_rate = max(0.0, min(1.0, to_float(cfg.get("min_prior_peak_cancel_rate", 0.80), 0.80)))
        min_cancel_drop = max(0.0, min(1.0, to_float(cfg.get("min_cancel_drop", 0.14), 0.14)))
        if cancel_rate_30s is not None and prior_peak_cancel_rate is not None:
            recovered = prior_peak_cancel_rate - cancel_rate_30s
            if prior_peak_cancel_rate >= min_prior_peak_cancel_rate and recovered >= min_cancel_drop:
                cancel_recovery = recovered
        if cancel_recovery is not None:
            edge += cancel_recovery * 5.0

        flow_imbalance = normalize_signed_ratio(row.get("flow_imbalance") or row.get("orderflow_imbalance"))
        if flow_imbalance is not None:
            flow_imbalance = abs(flow_imbalance)
        min_orderflow_alignment = max(0.0, min(1.0, to_float(cfg.get("min_orderflow_alignment", 0.05), 0.05)))
        if flow_imbalance is not None and flow_imbalance < min_orderflow_alignment:
            flow_imbalance = None
        if flow_imbalance is not None:
            edge += flow_imbalance * 1.5

        recent_move_zscore = safe_float(row.get("recent_move_zscore"), None)
        min_recent_move_zscore = max(0.0, to_float(cfg.get("min_recent_move_zscore", 1.25), 1.25))
        if recent_move_zscore is not None and abs(recent_move_zscore) < min_recent_move_zscore:
            recent_move_zscore = None
        if recent_move_zscore is not None:
            edge += min(3.0, abs(recent_move_zscore)) * 0.4

        min_edge_percent = max(0.0, to_float(cfg.get("min_edge_percent", 1.0), 1.0))
        if edge < min_edge_percent:
            return "edge_too_small"

        confidence = clamp(
            0.50
            + clamp(entropy - 0.5, 0, 0.20)
            + clamp(abs(diff_pct or 0) / 10.0, 0, 0.15)
            + clamp((1 - (spread or 0) / 0.065) * 0.08, 0, 0.08),
            0.40,
            0.92,
        )
        if cancel_recovery is not None:
            confidence = clamp(confidence + clamp(cancel_recovery * 0.12, 0.0, 0.08), 0.40, 0.92)
        if flow_imbalance is not None:
            confidence = clamp(confidence + clamp(flow_imbalance * 0.08, 0.0, 0.05), 0.40, 0.92)
        if recent_move_zscore is not None:
            confidence = clamp(confidence + clamp(abs(recent_move_zscore) / 6.0, 0.0, 0.04), 0.40, 0.92)

        min_confidence = to_confidence(cfg.get("min_confidence", 0.40), 0.40)
        if confidence < min_confidence:
            return "confidence_too_low"
        return "filtered"

    def _detect_from_rows(self, rows: list[dict[str, Any]]) -> list[Opportunity]:
        cfg = dict(self.default_config)
        cfg.update(getattr(self, "config", {}) or {})

        candidates: list[tuple[float, Opportunity]] = []
        rejection_counts: dict[str, int] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            signal = self._score_market(row, cfg)
            if signal is None:
                reason = self._rejection_reason(row, cfg)
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                continue
            firehose_gates = signal.get("_firehose_gates") or []
            opp = self._build_opportunity(row, signal)
            if opp is None:
                rejection_counts["invalid_opportunity"] = rejection_counts.get("invalid_opportunity", 0) + 1
                if firehose_gates:
                    firehose_gates.append(GateResult(
                        "build_opportunity", "Build opportunity", False,
                        detail="_build_opportunity returned None",
                    ))
                    emit_evaluation_nowait(
                        strategy_slug="crypto_entropy_maker",
                        market=row,
                        gates=firehose_gates,
                        outcome="rejected",
                        verbosity=MURMUR,
                    )
                continue
            emit_emit_nowait(
                strategy_slug="crypto_entropy_maker",
                market=row,
                detail=(
                    f"{signal['outcome']} @ {signal['entry_price']:.4f} • "
                    f"H={signal['entropy']:.3f} • edge={signal['edge']:.2f}% • "
                    f"conf={signal['confidence']:.2f}"
                ),
                extra={
                    "outcome": signal["outcome"],
                    "entry_price": signal["entry_price"],
                    "entropy": signal["entropy"],
                    "edge": signal["edge"],
                    "confidence": signal["confidence"],
                },
            )
            if firehose_gates:
                emit_evaluation_nowait(
                    strategy_slug="crypto_entropy_maker",
                    market=row,
                    gates=firehose_gates,
                    outcome="emitted",
                    verbosity=WHISPER,
                )
            candidates.append((float(signal["score"]), opp))

        top_rejections = sorted(rejection_counts.items(), key=lambda item: item[1], reverse=True)[:4]
        message_parts = [f"Scanned {len(rows)} markets, {len(candidates)} signals"]
        if top_rejections:
            message_parts.append(
                "rejected: " + ", ".join(f"{count} {reason}" for reason, count in top_rejections)
            )
        self._filter_diagnostics = {
            "strategy_key": self.strategy_type,
            "scanned_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "markets_scanned": len(rows),
            "signals_emitted": len(candidates),
            "rejections": rejection_counts,
            "message": " \u2014 ".join(message_parts),
            "summary": {f"rejected_{reason}": count for reason, count in rejection_counts.items()},
        }

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

    def evaluate(self, signal: Any, context: dict[str, Any]) -> StrategyDecision:
        params = context.get("params") or {}
        payload = signal_payload(signal)
        live_market = context.get("live_market")
        if not isinstance(live_market, dict):
            live_market = payload.get("live_market")
        if not isinstance(live_market, dict):
            live_market = {}

        min_edge = max(0.0, to_float(params.get("min_edge_percent", 1.0), 1.0))
        min_conf = to_confidence(params.get("min_confidence", 0.40), 0.40)
        min_entropy = max(0.0, min(1.0, to_float(params.get("min_entropy", 0.82), 0.82)))
        min_spread_pct = max(0.0, min(1.0, to_float(params.get("min_spread_pct", 0.006), 0.006)))
        max_spread_pct = max(min_spread_pct, min(1.0, to_float(params.get("max_spread_pct", 0.065), 0.065)))
        max_cancel_rate_30s = max(0.0, min(1.0, to_float(params.get("max_cancel_rate_30s", 0.75), 0.75)))
        min_liquidity_usd = max(0.0, to_float(params.get("min_liquidity_usd", 1000.0), 1000.0))

        base_size, max_size = _trader_size_limits(context)
        sizing_policy = str(params.get("sizing_policy", "adaptive") or "adaptive")

        source = str(getattr(signal, "source", "") or "").strip().lower()
        signal_type = str(getattr(signal, "signal_type", "") or "").strip().lower()
        direction = str(getattr(signal, "direction", "") or "").strip().lower()
        edge = max(0.0, to_float(getattr(signal, "edge_percent", 0.0), 0.0))
        confidence = to_confidence(getattr(signal, "confidence", 0.0), 0.0)
        liquidity = max(
            0.0,
            to_float(
                live_market.get("liquidity_usd"),
                to_float(payload.get("liquidity_usd"), to_float(getattr(signal, "liquidity", 0.0), 0.0)),
            ),
        )

        yes_price = safe_float(
            live_market.get("yes_price"),
            safe_float(live_market.get("live_yes_price"), safe_float(payload.get("yes_price"), safe_float(payload.get("up_price"), None))),
        )
        no_price = safe_float(
            live_market.get("no_price"),
            safe_float(live_market.get("live_no_price"), safe_float(payload.get("no_price"), safe_float(payload.get("down_price"), None))),
        )
        spread_pct = safe_float(live_market.get("spread"), safe_float(payload.get("spread"), None))
        cancel_rate_30s = normalize_ratio(
            live_market.get("cancel_rate_30s")
            if live_market.get("cancel_rate_30s") is not None
            else payload.get("cancel_rate_30s")
        )

        if yes_price is not None and no_price is not None:
            total = yes_price + no_price
            prob_yes = yes_price / total if total > 0 else 0.5
            entropy = _probability_entropy(prob_yes)
        else:
            entropy = None

        source_ok = source == "crypto"
        origin_ok = bool(payload.get("strategy_origin") == "crypto_worker") or signal_type.startswith("crypto_worker")
        edge_ok = edge >= min_edge
        confidence_ok = confidence >= min_conf
        entropy_ok = entropy is not None and entropy >= min_entropy
        spread_ok = spread_pct is not None and min_spread_pct <= spread_pct <= max_spread_pct
        cancel_ok = cancel_rate_30s is None or cancel_rate_30s <= max_cancel_rate_30s
        liquidity_ok = liquidity >= min_liquidity_usd

        checks = [
            DecisionCheck("source", "Crypto source", source_ok, detail="Requires source=crypto."),
            DecisionCheck("origin", "Crypto worker origin", origin_ok, detail="Requires crypto-worker signal."),
            DecisionCheck("edge", "Edge threshold", edge_ok, score=edge, detail=f"min={min_edge:.2f}%"),
            DecisionCheck(
                "confidence",
                "Confidence threshold",
                confidence_ok,
                score=confidence,
                detail=f"min={min_conf:.2f}",
            ),
            DecisionCheck(
                "entropy",
                "Binary entropy floor",
                entropy_ok,
                score=entropy,
                detail=f"min={min_entropy:.2f}",
            ),
            DecisionCheck(
                "spread_window",
                "Spread window",
                spread_ok,
                score=spread_pct,
                detail=f"range=[{min_spread_pct:.3f}, {max_spread_pct:.3f}]",
            ),
            DecisionCheck(
                "cancel_rate",
                "Cancel-rate cap",
                cancel_ok,
                score=cancel_rate_30s,
                detail=f"max={max_cancel_rate_30s:.2f}",
            ),
            DecisionCheck(
                "liquidity",
                "Liquidity floor",
                liquidity_ok,
                score=liquidity,
                detail=f"min={min_liquidity_usd:.0f}",
            ),
        ]

        score = (
            (edge * 0.45)
            + (confidence * 30.0)
            + (min(1.0, entropy or 0.0) * 10.0)
            + (min(1.0, liquidity / 20_000.0) * 4.0)
        )

        if not all(check.passed for check in checks):
            return StrategyDecision(
                decision="skipped",
                reason="Entropy-maker filters not met",
                score=score,
                checks=checks,
                payload={
                    "edge": edge,
                    "confidence": confidence,
                    "entropy": entropy,
                    "spread_pct": spread_pct,
                    "cancel_rate_30s": cancel_rate_30s,
                },
            )

        probability = selected_probability(signal, payload, direction)
        entry_price = to_float(getattr(signal, "entry_price", 0.0), 0.0)
        if entry_price <= 0.0:
            entry_price = to_float(live_market.get("live_selected_price"), to_float(payload.get("selected_price"), 0.0))
        entropy_multiplier = 0.55 + (0.80 * float(entropy or 0.0))

        sizing = compute_position_size(
            base_size_usd=base_size * entropy_multiplier,
            max_size_usd=max_size,
            edge_percent=edge,
            confidence=confidence,
            sizing_policy=sizing_policy,
            probability=probability,
            entry_price=entry_price if entry_price > 0 else None,
            liquidity_usd=liquidity,
            liquidity_cap_fraction=0.08,
            min_size_usd=1.0,
        )
        size_usd = float(sizing.get("size_usd") or base_size)

        return StrategyDecision(
            decision="selected",
            reason="Entropy-maker signal selected",
            score=score,
            size_usd=size_usd,
            checks=checks,
            payload={
                "edge": edge,
                "confidence": confidence,
                "entropy": entropy,
                "entropy_multiplier": entropy_multiplier,
                "spread_pct": spread_pct,
                "cancel_rate_30s": cancel_rate_30s,
                "sizing": sizing,
            },
        )

    def should_exit(self, position: Any, market_state: dict) -> ExitDecision:
        config = dict(getattr(position, "config", None) or {})
        strategy_config = dict(getattr(self, "config", None) or {})
        config.setdefault("take_profit_pct", float(strategy_config.get("take_profit_pct", 6.5)))
        config.setdefault("stop_loss_pct", float(strategy_config.get("stop_loss_pct", 4.0)))
        config.setdefault("max_hold_minutes", float(strategy_config.get("max_hold_minutes", 16.0)))
        position.config = config
        return self.default_exit_check(position, market_state)

    def on_blocked(self, signal, reason: str, context: dict) -> None:
        logger.info("%s blocked: %s market=%s", self.name, reason, getattr(signal, "market_id", "?"))
