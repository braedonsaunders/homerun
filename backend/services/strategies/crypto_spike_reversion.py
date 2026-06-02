# SEED TEMPLATE — not the live strategy. The DB `strategies.source_code` is the
# runtime master (loaded by the orchestrator + backtester); edits here only seed
# fresh installs / reset-to-factory. See services/opportunity_strategy_catalog.py.
"""Crypto spike-reversion strategy.

Detects opportunities directly from ``crypto_update`` worker payloads where a
sharp short-horizon price spike has occurred and a mean-reversion entry is
favorable.  Uses 5m/30m/2h price movement context to validate direction
alignment and reversion shape before sizing with Kelly or other configurable
policies.
"""

from __future__ import annotations

import logging
from typing import Any

from utils.utcnow import utcnow  # replay-clock-aware "now" (honors backtest sim time)
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
    parse_datetime_utc,
    pick_oracle_source,
    polymarket_taker_fee_pct,
    timeframe_seconds,
)
from services.strategies.reversion_helpers import direction_opposes_impulse, market_move_pct, reversion_shape_ok
from services.trader_orchestrator.strategies.sizing import compute_position_size
from utils.converters import clamp, safe_float, to_confidence, to_float
from utils.signal_helpers import selected_probability, signal_payload

logger = logging.getLogger(__name__)


class CryptoSpikeReversionStrategy(BaseStrategy):
    """Spike-reversion detection and execution using live 5m/30m/2h movement context."""

    strategy_type = "crypto_spike_reversion"
    name = "Crypto Spike Reversion"
    description = "Spike-reversion detection using live 5m/30m/2h movement context"
    mispricing_type = "within_market"
    source_key = "crypto"
    market_categories = ["crypto"]
    subscriptions = [EventType.CRYPTO_UPDATE]
    supports_entry_take_profit_exit = True
    default_open_order_timeout_seconds = 20.0

    quality_filter_overrides = QualityFilterOverrides(
        min_roi=1.0,
        max_resolution_months=0.1,
    )

    default_config = {
        "min_edge_percent": 2.8,
        "min_confidence": 0.44,
        "min_abs_move_5m": 1.8,
        "max_abs_move_2h": 14.0,
        "require_reversion_shape": True,
        "min_order_size_usd": 2.0,
        "sizing_policy": "kelly",
        "kelly_fractional_scale": 0.45,
        "take_profit_pct": 8.0,
        "stop_loss_pct": 4.0,
        "max_hold_minutes": 8.0,
        "liquidity_cap_fraction": 0.07,
        "min_liquidity_usd": 2000.0,
        "max_entry_price": 0.92,
        "max_markets_per_event": 24,
        # Safety gates — leave at None to use the timeframe-aware defaults in
        # crypto_strategy_utils. Override per-strategy in the UI to tighten or
        # loosen for specific markets.
        "max_oracle_age_ms": None,
        "max_market_data_age_ms": None,
        "min_seconds_left_for_entry": None,
        "min_fee_clearance_x": 2.0,
        "prefer_oracle_source": "binance_direct",
        # Force exit when this close to resolution. Spike uses IOC so the
        # entry path won't rest, but a held position can still ride into
        # the heartbeat window — close it with a market sell first.
        "force_exit_seconds_before_resolution": 20.0,
    }

    def __init__(self) -> None:
        super().__init__()
        self.min_profit = 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_market(row: dict[str, Any]) -> Market | None:
        return build_binary_crypto_market(row)

    # ------------------------------------------------------------------
    # Signal scoring — spike-reversion specific
    # ------------------------------------------------------------------

    def _score_market(self, row: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any] | None:
        gates: list[GateResult] = []

        def _emit_reject(verbosity: str = MURMUR) -> None:
            emit_evaluation_nowait(
                strategy_slug="crypto_spike_reversion",
                market=row,
                gates=gates,
                outcome="rejected",
                verbosity=verbosity,
            )

        up_price = safe_float(row.get("up_price"), None)
        down_price = safe_float(row.get("down_price"), None)
        prices_ok = up_price is not None and down_price is not None
        gates.append(GateResult(
            "prices_present", "Up/down prices present", prices_ok,
            detail=f"up={up_price} down={down_price}",
        ))
        if not prices_ok:
            _emit_reject(WHISPER)
            return None

        move_5m = safe_float(row.get("move_5m_percent"), safe_float(row.get("move_5m_pct"), None))
        move_30m = safe_float(row.get("move_30m_percent"), safe_float(row.get("move_30m_pct"), None))
        move_2h = safe_float(row.get("move_2h_percent"), safe_float(row.get("move_2h_pct"), None))

        move_5m_ok = move_5m is not None
        gates.append(GateResult(
            "move_5m_present", "5m move present", move_5m_ok,
            detail=f"move_5m={move_5m}",
        ))
        if not move_5m_ok:
            _emit_reject(WHISPER)
            return None

        min_abs_move_5m = max(0.0, to_float(cfg.get("min_abs_move_5m", 1.8), 1.8))
        max_abs_move_2h = max(min_abs_move_5m, to_float(cfg.get("max_abs_move_2h", 14.0), 14.0))
        require_reversion_shape = bool(cfg.get("require_reversion_shape", True))

        move_5m_threshold_ok = abs(move_5m) >= min_abs_move_5m
        gates.append(GateResult(
            "move_5m_threshold", "|5m move| >= threshold", move_5m_threshold_ok,
            score=float(move_5m),
            detail=f"|move_5m|={abs(move_5m):.3f}% min={min_abs_move_5m:.3f}%",
        ))
        if not move_5m_threshold_ok:
            _emit_reject(MURMUR)
            return None

        # Reversion shape: short-horizon impulse dominates the 30m trend
        shape_ok = reversion_shape_ok(
            move_5m,
            move_30m,
            move_2h,
            require_shape=require_reversion_shape,
            max_abs_move_2h=max_abs_move_2h,
        )

        shape_gate_passed = (not require_reversion_shape) or shape_ok
        gates.append(GateResult(
            "reversion_shape", "Reversion shape valid", shape_gate_passed,
            detail=f"shape_ok={shape_ok} require={require_reversion_shape} 30m={move_30m} 2h={move_2h} max_2h={max_abs_move_2h:.2f}",
        ))
        if require_reversion_shape and not shape_ok:
            _emit_reject(MURMUR)
            return None

        # Direction OPPOSES the spike: spike up -> buy_no, spike down -> buy_yes
        if move_5m > 0:
            direction = "buy_no"
            outcome = "NO"
            selected_price = float(down_price)
        else:
            direction = "buy_yes"
            outcome = "YES"
            selected_price = float(up_price)

        max_entry_price = clamp(to_float(cfg.get("max_entry_price", 0.92), 0.92), 0.05, 0.99)
        entry_price_ok = 0.0 < selected_price < 1.0 and selected_price <= max_entry_price
        gates.append(GateResult(
            "entry_price", "Entry price within bounds", entry_price_ok,
            score=selected_price,
            detail=f"price={selected_price:.4f} max={max_entry_price:.4f} outcome={outcome}",
        ))
        if not entry_price_ok:
            _emit_reject(MURMUR)
            return None

        # ---- Resolution-boundary safety: refuse to enter when there's not
        # enough runway after the fill to react before the Chainlink heartbeat
        # crosses the resolution timestamp.
        timeframe_value = row.get("timeframe")
        tf_seconds = timeframe_seconds(timeframe_value)
        end_date = parse_datetime_utc(row.get("end_time"))
        seconds_left = float("inf")
        elapsed_ratio = 0.5  # default mid-window
        if end_date is not None:
            seconds_left = max(0.0, (end_date - utcnow()).total_seconds())
            elapsed_ratio = clamp(1.0 - (seconds_left / max(1, tf_seconds)), 0.0, 1.0)

        configured_min_secs = cfg.get("min_seconds_left_for_entry")
        min_seconds_left = (
            float(configured_min_secs)
            if configured_min_secs is not None
            else default_min_seconds_left_for_entry(timeframe_value)
        )
        seconds_left_ok = seconds_left >= min_seconds_left
        gates.append(GateResult(
            "min_seconds_left", "Seconds left for entry", seconds_left_ok,
            score=float(seconds_left) if seconds_left != float("inf") else None,
            detail=f"left={seconds_left:.1f}s min={min_seconds_left:.1f}s",
        ))
        if not seconds_left_ok:
            _emit_reject(MURMUR)
            return None

        # ---- Worker-snapshot freshness: avoid trading on a stale row.
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

        # ---- Oracle: prefer the freshest source (typically binance_direct
        # sub-second WS) and refuse if every source is past the per-timeframe
        # staleness cap. The Chainlink-dominated row.get("oracle_price") is
        # left as a fallback only.
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
            # Fall back to the row's primary price only when its own age is
            # within the cap; otherwise treat oracle as unavailable.
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

        # Edge: base on spike magnitude; add oracle component if available
        if oracle_price is not None and price_to_beat is not None and price_to_beat > 0:
            diff_pct = abs(((oracle_price - price_to_beat) / price_to_beat) * 100.0)
            edge = abs(move_5m) * 0.6 + diff_pct
        else:
            diff_pct = 0.0
            edge = abs(move_5m) * 0.6

        confidence = clamp(
            0.50
            + clamp(abs(move_5m) / 12.0, 0, 0.20)
            + (0.10 if shape_ok else 0)
            + clamp(elapsed_ratio * 0.10, 0, 0.10),
            0.44,
            0.90,
        )

        min_confidence = to_confidence(cfg.get("min_confidence", 0.44), 0.44)
        confidence_ok = confidence >= min_confidence
        gates.append(GateResult(
            "min_confidence", "Min confidence", confidence_ok,
            score=float(confidence),
            detail=f"conf={confidence:.3f} min={min_confidence:.3f}",
        ))
        if not confidence_ok:
            _emit_reject(MURMUR)
            return None

        liquidity = max(0.0, float(safe_float(row.get("liquidity"), 0.0) or 0.0))
        min_liquidity_usd = max(0.0, to_float(cfg.get("min_liquidity_usd", 2000.0), 2000.0))
        liquidity_ok = liquidity >= min_liquidity_usd
        gates.append(GateResult(
            "min_liquidity", "Min liquidity", liquidity_ok,
            score=float(liquidity),
            detail=f"liq=${liquidity:.0f} min=${min_liquidity_usd:.0f}",
        ))
        if not liquidity_ok:
            _emit_reject(MURMUR)
            return None

        risk_score = clamp(
            0.60
            - (min(1.0, abs(move_5m) / 8.0) * 0.20)
            - (0.08 if shape_ok else 0)
            + (0.10 if diff_pct < 0.1 else 0),
            0.15,
            0.85,
        )

        score = (
            (edge * 0.55)
            + (confidence * 34.0)
            + (min(1.0, abs(move_5m) / 8.0) * 8.0)
            + (min(1.0, liquidity / 20000.0) * 4.0)
        )

        # ---- Fee-aware net edge: use the docs-accurate Polymarket taker
        # curve (`p * 0.25 * (p*(1-p))^2`) rather than a flat 0.25% guess.
        # At p=0.30 the real fee is ~1.10%, at p=0.50 it's ~1.56%, so the old
        # constant was 4-6× too low at typical entry prices and would let
        # marginal trades through with negative true edge.
        taker_fee_pct_value = polymarket_taker_fee_pct(selected_price) * 100.0
        net_edge_percent = max(0.0, edge - taker_fee_pct_value)

        # Hard fee-clearance gate: refuse trades whose raw edge can't clear
        # the taker fee by ``min_fee_clearance_x``×.  Floor relaxed to 0.0
        # so a config of 0 truly disables this gate.
        min_fee_clearance_x = max(0.0, to_float(cfg.get("min_fee_clearance_x", 2.0), 2.0))
        min_edge_required = fee_aware_min_edge_pct(selected_price, min_fee_clearance_x)
        fee_clearance_ok = edge >= min_edge_required
        gates.append(GateResult(
            "fee_clearance", "Edge clears fee threshold", fee_clearance_ok,
            score=float(edge),
            detail=f"edge={edge:.3f}% min={min_edge_required:.3f}% (taker_fee={taker_fee_pct_value:.3f}% × {min_fee_clearance_x:.1f})",
        ))
        if not fee_clearance_ok:
            _emit_reject(MURMUR)
            return None

        target_exit_price = clamp(selected_price + (net_edge_percent / 100.0), selected_price + 0.0001, 0.999)

        # All gates in _score_market passed.  Record success here; the
        # final VOICE-tier emit fires when _build_opportunity actually
        # produces an Opportunity.
        gates.append(GateResult(
            "score_market_all", "All scoring gates passed", True,
            score=float(edge),
            detail=f"net_edge={net_edge_percent:.3f}% conf={confidence:.3f}",
        ))

        return {
            "_firehose_gates": list(gates),
            "direction": direction,
            "outcome": outcome,
            "selected_price": selected_price,
            "move_5m": float(move_5m),
            "move_30m": float(move_30m) if move_30m is not None else None,
            "move_2h": float(move_2h) if move_2h is not None else None,
            "oracle_price": float(oracle_price) if oracle_price is not None else None,
            "oracle_source_used": oracle_source_used,
            "oracle_age_ms": float(oracle_age_ms) if oracle_age_ms is not None else None,
            "price_to_beat": float(price_to_beat) if price_to_beat is not None else None,
            "oracle_diff_pct": float(diff_pct),
            "edge": float(edge),
            "taker_fee_pct": float(taker_fee_pct_value),
            "net_edge_percent": float(net_edge_percent),
            "confidence": float(confidence),
            "risk_score": float(risk_score),
            "shape_ok": shape_ok,
            "elapsed_ratio": float(elapsed_ratio),
            "seconds_left": float(seconds_left) if seconds_left != float("inf") else None,
            "timeframe": str(timeframe_value or "") or None,
            "liquidity_usd": float(liquidity),
            "target_exit_price": float(target_exit_price),
            "score": float(score),
        }

    # ------------------------------------------------------------------
    # Opportunity builder
    # ------------------------------------------------------------------

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

        reversion_context = {
            "strategy_origin": "crypto_worker",
            "direction": direction,
            "move_5m": signal["move_5m"],
            "move_30m": signal["move_30m"],
            "move_2h": signal["move_2h"],
            "oracle_price": signal["oracle_price"],
            "oracle_source_used": signal.get("oracle_source_used"),
            "oracle_age_ms": signal.get("oracle_age_ms"),
            "price_to_beat": signal["price_to_beat"],
            "oracle_diff_pct": signal["oracle_diff_pct"],
            "edge": signal["edge"],
            "taker_fee_pct": signal.get("taker_fee_pct"),
            "net_edge_percent": signal["net_edge_percent"],
            "confidence": signal["confidence"],
            "risk_score": signal["risk_score"],
            "shape_ok": signal["shape_ok"],
            "elapsed_ratio": signal["elapsed_ratio"],
            "seconds_left": signal.get("seconds_left"),
            "timeframe": signal.get("timeframe"),
            "liquidity_usd": signal["liquidity_usd"],
            "target_exit_price": signal["target_exit_price"],
            "market_data_age_ms": safe_float(row.get("market_data_age_ms"), None),
            "fetched_at": row.get("fetched_at"),
            "start_time": row.get("start_time"),
            "end_time": row.get("end_time"),
            "regime": row.get("regime"),
            "regime_params": row.get("regime_params"),
        }

        positions = [
            {
                "action": "BUY",
                "outcome": outcome,
                "price": selected_price,
                "token_id": token_id,
                "price_policy": "taker_limit",
                "time_in_force": "IOC",
                "_reversion": reversion_context,
            }
        ]

        opp = self.create_opportunity(
            title=f"Spike Reversion: {outcome} @ {selected_price:.3f} (5m={signal['move_5m']:+.2f}%)",
            description=(
                f"Mean-reversion after {signal['move_5m']:+.2f}% 5m spike, "
                f"edge {signal['edge']:.2f}%, conf {signal['confidence']:.2f}"
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
            min_liquidity_hard=max(100.0, to_float(strategy_cfg.get("min_liquidity_usd", 2000.0), 2000.0)),
            min_position_size=max(1.0, to_float(strategy_cfg.get("min_order_size_usd", 2.0), 2.0)),
        )
        if opp is None:
            return None

        opp.risk_factors = [
            f"Spike reversion (5m={signal['move_5m']:+.2f}%)",
            f"Shape {'valid' if signal['shape_ok'] else 'weak'} (30m={signal['move_30m']}, 2h={signal['move_2h']})",
            f"Edge={signal['edge']:.2f}%, net={signal['net_edge_percent']:.2f}%",
        ]
        opp.strategy_context = {
            "source_key": "crypto",
            "strategy_slug": self.strategy_type,
            **reversion_context,
        }
        return opp

    def _rejection_reason(self, row: dict[str, Any], cfg: dict[str, Any]) -> str:
        up_price = safe_float(row.get("up_price"), None)
        down_price = safe_float(row.get("down_price"), None)
        if up_price is None or down_price is None:
            return "missing_prices"

        move_5m = safe_float(row.get("move_5m_percent"), safe_float(row.get("move_5m_pct"), None))
        move_30m = safe_float(row.get("move_30m_percent"), safe_float(row.get("move_30m_pct"), None))
        move_2h = safe_float(row.get("move_2h_percent"), safe_float(row.get("move_2h_pct"), None))
        if move_5m is None:
            return "missing_move_5m"

        min_abs_move_5m = max(0.0, to_float(cfg.get("min_abs_move_5m", 1.8), 1.8))
        if abs(move_5m) < min_abs_move_5m:
            return "move_below_threshold"

        max_abs_move_2h = max(min_abs_move_5m, to_float(cfg.get("max_abs_move_2h", 14.0), 14.0))
        require_reversion_shape = bool(cfg.get("require_reversion_shape", True))
        shape_ok = reversion_shape_ok(
            move_5m,
            move_30m,
            move_2h,
            require_shape=require_reversion_shape,
            max_abs_move_2h=max_abs_move_2h,
        )
        if require_reversion_shape and not shape_ok:
            return "shape_invalid"

        selected_price = float(down_price if move_5m > 0 else up_price)
        max_entry_price = clamp(to_float(cfg.get("max_entry_price", 0.92), 0.92), 0.05, 0.99)
        if selected_price <= 0.0 or selected_price >= 1.0:
            return "invalid_entry_price"
        if selected_price > max_entry_price:
            return "entry_price_too_high"

        oracle_price = safe_float(row.get("oracle_price"), None)
        price_to_beat = safe_float(row.get("price_to_beat"), None)
        if oracle_price is not None and price_to_beat is not None and price_to_beat > 0:
            diff_pct = abs(((oracle_price - price_to_beat) / price_to_beat) * 100.0)
            edge = abs(move_5m) * 0.6 + diff_pct
        else:
            diff_pct = 0.0
            edge = abs(move_5m) * 0.6

        end_date = parse_datetime_utc(row.get("end_time"))
        elapsed_ratio = 0.5
        if end_date is not None:
            seconds_left = max(0.0, (end_date - utcnow()).total_seconds())
            elapsed_ratio = clamp(1.0 - (seconds_left / 300.0), 0.0, 1.0)
        confidence = clamp(
            0.50
            + clamp(abs(move_5m) / 12.0, 0, 0.20)
            + (0.10 if shape_ok else 0)
            + clamp(elapsed_ratio * 0.10, 0, 0.10),
            0.44,
            0.90,
        )
        min_confidence = to_confidence(cfg.get("min_confidence", 0.44), 0.44)
        if confidence < min_confidence:
            return "confidence_too_low"

        liquidity = max(0.0, float(safe_float(row.get("liquidity"), 0.0) or 0.0))
        min_liquidity_usd = max(0.0, to_float(cfg.get("min_liquidity_usd", 2000.0), 2000.0))
        if liquidity < min_liquidity_usd:
            return "low_liquidity"

        min_edge_percent = max(0.0, to_float(cfg.get("min_edge_percent", 2.8), 2.8))
        if edge < min_edge_percent:
            return "edge_too_small"
        return "filtered"

    # ------------------------------------------------------------------
    # Detection from raw crypto rows
    # ------------------------------------------------------------------

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
                        strategy_slug="crypto_spike_reversion",
                        market=row,
                        gates=firehose_gates,
                        outcome="rejected",
                        verbosity=MURMUR,
                    )
                continue
            # Opportunity emitted — VOICE tier.
            emit_emit_nowait(
                strategy_slug="crypto_spike_reversion",
                market=row,
                detail=(
                    f"{signal['outcome']} @ {signal['selected_price']:.4f} • "
                    f"5m={signal['move_5m']:+.2f}% • edge={signal['edge']:.2f}% • "
                    f"net={signal['net_edge_percent']:.2f}% • conf={signal['confidence']:.2f}"
                ),
                extra={
                    "outcome": signal["outcome"],
                    "selected_price": signal["selected_price"],
                    "move_5m": signal["move_5m"],
                    "edge": signal["edge"],
                    "net_edge_percent": signal["net_edge_percent"],
                    "confidence": signal["confidence"],
                },
            )
            if firehose_gates:
                emit_evaluation_nowait(
                    strategy_slug="crypto_spike_reversion",
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
            "scanned_at": utcnow().replace(microsecond=0).isoformat().replace("+00:00", "Z"),
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

    # ------------------------------------------------------------------
    # Event handler — now generates its own signals
    # ------------------------------------------------------------------

    async def on_event(self, event: DataEvent) -> list[Opportunity]:
        if event.event_type != EventType.CRYPTO_UPDATE:
            return []
        rows = event.payload.get("markets") if isinstance(event.payload, dict) else None
        if not isinstance(rows, list) or not rows:
            return []
        return self._detect_from_rows(rows)

    # ------------------------------------------------------------------
    # Evaluate (unchanged)
    # ------------------------------------------------------------------

    def evaluate(self, signal: Any, context: dict[str, Any]) -> StrategyDecision:
        params = context.get("params") or {}
        payload = signal_payload(signal)
        live_market = context.get("live_market")
        if not isinstance(live_market, dict):
            live_market = payload.get("live_market")
        if not isinstance(live_market, dict):
            live_market = {}

        min_edge = to_float(params.get("min_edge_percent", 2.8), 2.8)
        min_conf = to_confidence(params.get("min_confidence", 0.44), 0.44)
        min_abs_move_5m = max(0.2, to_float(params.get("min_abs_move_5m", 1.8), 1.8))
        max_abs_move_2h = max(min_abs_move_5m, to_float(params.get("max_abs_move_2h", 14.0), 14.0))
        require_reversion_shape = bool(params.get("require_reversion_shape", True))

        base_size, max_size = _trader_size_limits(context)
        sizing_policy = str(params.get("sizing_policy", "kelly") or "kelly")
        kelly_fractional_scale = to_float(params.get("kelly_fractional_scale", 0.45), 0.45)

        source = str(getattr(signal, "source", "") or "").strip().lower()
        direction = str(getattr(signal, "direction", "") or "").strip().lower()
        signal_type = str(getattr(signal, "signal_type", "") or "").strip().lower()

        edge = max(0.0, to_float(getattr(signal, "edge_percent", 0.0), 0.0))
        confidence = to_confidence(getattr(signal, "confidence", 0.0), 0.0)
        liquidity = max(0.0, to_float(getattr(signal, "liquidity", 0.0), 0.0))

        move_5m = market_move_pct(live_market, payload, "move_5m")
        move_30m = market_move_pct(live_market, payload, "move_30m")
        move_2h = market_move_pct(live_market, payload, "move_2h")

        # Direction alignment: signal direction must oppose the spike
        direction_alignment = direction_opposes_impulse(direction, move_5m, min_abs_move_5m)

        # Reversion shape: short-horizon impulse dominates the 30m trend
        shape_ok = reversion_shape_ok(
            move_5m,
            move_30m,
            move_2h,
            require_shape=require_reversion_shape,
            max_abs_move_2h=max_abs_move_2h,
        )

        origin_ok = bool(payload.get("strategy_origin") == "crypto_worker") or signal_type.startswith("crypto_worker")

        checks = [
            DecisionCheck("source", "Crypto source", source == "crypto", detail="Requires source=crypto."),
            DecisionCheck(
                "origin", "Crypto worker origin", origin_ok, detail="Requires worker-generated crypto signal."
            ),
            DecisionCheck("edge", "Edge threshold", edge >= min_edge, score=edge, detail=f"min={min_edge:.2f}"),
            DecisionCheck(
                "confidence",
                "Confidence threshold",
                confidence >= min_conf,
                score=confidence,
                detail=f"min={min_conf:.2f}",
            ),
            DecisionCheck(
                "direction_alignment",
                "Direction aligns with spike",
                direction_alignment,
                score=move_5m,
                detail=f"|move_5m| >= {min_abs_move_5m:.2f}% and opposes impulse",
            ),
            DecisionCheck(
                "reversion_shape",
                "Reversion shape",
                shape_ok,
                score=move_30m,
                detail=f"2h cap <= {max_abs_move_2h:.2f}%",
            ),
        ]

        score = (
            (edge * 0.55)
            + (confidence * 34.0)
            + (min(1.0, abs(move_5m or 0.0) / 8.0) * 8.0)
            + (min(1.0, liquidity / 20000.0) * 4.0)
        )

        if not all(check.passed for check in checks):
            return StrategyDecision(
                decision="skipped",
                reason="Crypto spike-reversion filters not met",
                score=score,
                checks=checks,
                payload={
                    "edge": edge,
                    "confidence": confidence,
                    "move_5m": move_5m,
                    "move_30m": move_30m,
                    "move_2h": move_2h,
                },
            )

        probability = selected_probability(signal, payload, direction)
        entry_price = to_float(getattr(signal, "entry_price", 0.0), 0.0)

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
            liquidity_cap_fraction=0.07,
        )

        return StrategyDecision(
            decision="selected",
            reason="Crypto spike-reversion signal selected",
            score=score,
            size_usd=float(sizing["size_usd"]),
            checks=checks,
            payload={
                "edge": edge,
                "confidence": confidence,
                "move_5m": move_5m,
                "move_30m": move_30m,
                "move_2h": move_2h,
                "sizing": sizing,
            },
        )

    # ------------------------------------------------------------------
    # Exit
    # ------------------------------------------------------------------

    def should_exit(self, position: Any, market_state: dict) -> ExitDecision:
        """Tight exit for spike reversion -- short hold with explicit TP default."""
        config = getattr(position, "config", None) or {}
        config = dict(config)
        configured_tp = (getattr(self, "config", None) or {}).get("take_profit_pct", 8.0)
        try:
            default_tp = float(configured_tp)
        except (TypeError, ValueError):
            default_tp = 8.0
        config.setdefault("take_profit_pct", default_tp)
        position.config = config
        return self.default_exit_check(position, market_state)

    # ------------------------------------------------------------------
    # Platform gate hooks
    # ------------------------------------------------------------------

    def on_blocked(self, signal, reason: str, context: dict) -> None:
        logger.info("%s: signal blocked — %s (market=%s)", self.name, reason, getattr(signal, "market_id", "?"))

    def on_size_capped(self, original_size: float, capped_size: float, reason: str) -> None:
        logger.info("%s: size capped $%.0f → $%.0f — %s", self.name, original_size, capped_size, reason)
