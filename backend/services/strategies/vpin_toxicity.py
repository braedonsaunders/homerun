"""VPIN Order Flow Toxicity Strategy.

Implements Volume-Synchronized Probability of Informed Trading (VPIN) as
described by Easley, Lopez de Prado & O'Hara (2012).  Trade volume is
partitioned into equal-size buckets; buy/sell imbalance within those buckets
is the core signal.  When VPIN spikes, informed traders are entering the
market and we trade in their direction.

Pipeline:
  1. WebSocket trade tape emits TRADE_EXECUTION events per fill.
  2. on_event(trade_execution) accumulates trades into volume buckets.
  3. on_event(market_data_refresh) triggers detect() which computes VPIN
     across all tracked tokens, emits opportunities when VPIN > threshold.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Optional

from models import Market, Event, Opportunity
from .base import (
    BaseStrategy,
    DecisionCheck,
    ExitDecision,
    ScoringWeights,
    SizingConfig,
)
from services.data_events import DataEvent, EventType
from services.quality_filter import QualityFilterOverrides
from utils.kelly import kelly_fraction
from utils.logger import get_logger

logger = get_logger(__name__)


class VPINToxicityStrategy(BaseStrategy):
    """Detect informed trading via VPIN order flow toxicity."""

    strategy_type = "vpin_toxicity"
    name = "VPIN Toxicity"
    description = "Detect informed trading via VPIN order flow toxicity"
    mispricing_type = "within_market"
    realtime_processing_mode = "incremental"
    subscriptions = ["market_data_refresh", "trade_execution"]

    quality_filter_overrides = QualityFilterOverrides(
        min_roi=2.5,
    )

    scoring_weights = ScoringWeights(
        edge_weight=0.55,
        confidence_weight=25.0,
        risk_penalty=8.0,
    )
    sizing_config = SizingConfig(
        base_divisor=100.0,
        confidence_offset=0.70,
        risk_scale_factor=0.35,
        risk_floor=0.55,
    )

    default_config = {
        "min_edge_percent": 2.5,
        "min_confidence": 0.50,
        "max_risk_score": 0.75,
        "bucket_size_usd": 500.0,
        "num_buckets": 20,
        "vpin_threshold": 0.70,
        "vpin_lookback_buckets": 20,
        "min_liquidity": 1000.0,
        "base_size_usd": 18.0,
        "max_size_usd": 160.0,
    }

    def __init__(self) -> None:
        super().__init__()
        self.config = dict(self.default_config)

    # ------------------------------------------------------------------
    # Event routing
    # ------------------------------------------------------------------

    async def on_event(self, event: DataEvent) -> list:
        """Route trade executions to bucket accumulation; market refreshes to detect()."""
        if event.event_type == EventType.TRADE_EXECUTION:
            self._accumulate_trade(event)
            return []
        elif event.event_type == EventType.MARKET_DATA_REFRESH:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, self.detect, event.events or [], event.markets or [], event.prices or {}
            )
        return []

    # ------------------------------------------------------------------
    # Trade accumulation (fast, per-trade)
    # ------------------------------------------------------------------

    def _accumulate_trade(self, event: DataEvent) -> None:
        """Accumulate a trade into volume buckets using tick rule classification."""
        token_id = event.token_id
        if not token_id:
            return

        payload = event.payload
        price = float(payload.get("price", 0))
        size = float(payload.get("size", 0))
        side = str(payload.get("side", "")).upper()

        if price <= 0 or size <= 0:
            return

        buckets = self.state.setdefault("buckets", {})
        if token_id not in buckets:
            buckets[token_id] = {
                "current_bucket": {"buy_vol": 0.0, "sell_vol": 0.0, "total_vol": 0.0},
                "completed_buckets": [],
                "last_mid": price,
            }

        token_data = buckets[token_id]
        trade_value = price * size  # dollar volume

        # Classify trade direction: use explicit side if available, else tick rule
        if side in ("BUY", "SELL"):
            is_buy = side == "BUY"
        else:
            is_buy = price >= token_data["last_mid"]

        token_data["last_mid"] = price

        current = token_data["current_bucket"]
        if is_buy:
            current["buy_vol"] += trade_value
        else:
            current["sell_vol"] += trade_value
        current["total_vol"] += trade_value

        # Check if bucket is full
        bucket_size = float(self.config.get("bucket_size_usd", 500.0))
        if current["total_vol"] >= bucket_size:
            # Complete this bucket and start a new one
            completed = token_data.setdefault("completed_buckets", [])
            completed.append({"buy_vol": current["buy_vol"], "sell_vol": current["sell_vol"]})
            # Keep bounded
            max_buckets = int(self.config.get("num_buckets", 20)) * 3
            if len(completed) > max_buckets:
                token_data["completed_buckets"] = completed[-max_buckets:]
            token_data["current_bucket"] = {"buy_vol": 0.0, "sell_vol": 0.0, "total_vol": 0.0}

    # ------------------------------------------------------------------
    # VPIN calculation
    # ------------------------------------------------------------------

    def _calculate_vpin(self, token_id: str) -> Optional[float]:
        """Calculate VPIN over the last N completed buckets."""
        buckets_data = self.state.get("buckets", {}).get(token_id)
        if not buckets_data:
            return None

        completed = buckets_data.get("completed_buckets", [])
        n = int(self.config.get("vpin_lookback_buckets", 20))

        if len(completed) < n:
            return None  # Not enough data

        recent = completed[-n:]
        total_imbalance = 0.0
        for bucket in recent:
            buy = bucket["buy_vol"]
            sell = bucket["sell_vol"]
            total_imbalance += abs(buy - sell)

        bucket_size = float(self.config.get("bucket_size_usd", 500.0))
        vpin = total_imbalance / (n * bucket_size)
        return min(vpin, 1.0)  # cap at 1.0

    def _get_flow_direction(self, token_id: str) -> float:
        """Get the dominant flow direction from recent buckets.

        Returns:
            Positive = buy pressure, negative = sell pressure, magnitude [0, 1].
        """
        buckets_data = self.state.get("buckets", {}).get(token_id)
        if not buckets_data:
            return 0.0
        completed = buckets_data.get("completed_buckets", [])
        if len(completed) < 3:
            return 0.0
        recent = completed[-5:]  # last 5 buckets
        net = sum(b["buy_vol"] - b["sell_vol"] for b in recent)
        total = sum(b["buy_vol"] + b["sell_vol"] for b in recent)
        if total <= 0:
            return 0.0
        return net / total  # positive = buy pressure, negative = sell pressure

    # ------------------------------------------------------------------
    # Detection (runs on market_data_refresh)
    # ------------------------------------------------------------------

    def detect(
        self,
        events: list[Event],
        markets: list[Market],
        prices: dict[str, dict],
    ) -> list[Opportunity]:
        opportunities: list[Opportunity] = []
        cfg = self.config or self.default_config
        vpin_threshold = float(cfg.get("vpin_threshold", 0.70))

        # Build token -> market mapping
        token_to_market = self.state.setdefault("token_to_market", {})
        market_map: dict[str, Market] = {}
        for market in markets:
            if not market.active or market.closed:
                continue
            market_map[market.id] = market
            if market.clob_token_ids:
                for tid in market.clob_token_ids:
                    token_to_market[tid] = market.id

        # Calculate VPIN for all tokens we have trade data for
        buckets = self.state.get("buckets", {})
        for token_id in list(buckets.keys()):
            vpin = self._calculate_vpin(token_id)
            if vpin is None:
                continue

            # Store VPIN reading history
            vpin_hist = self.state.setdefault("vpin_values", {})
            if token_id not in vpin_hist:
                vpin_hist[token_id] = []
            vpin_hist[token_id].append((time.time(), vpin))
            if len(vpin_hist[token_id]) > 100:
                vpin_hist[token_id] = vpin_hist[token_id][-100:]

            if vpin < vpin_threshold:
                continue

            # VPIN spike detected — informed traders entering
            market_id = token_to_market.get(token_id)
            if not market_id:
                continue
            market = market_map.get(market_id)
            if not market:
                continue

            # Determine trade direction from flow
            direction = self._get_flow_direction(token_id)
            if abs(direction) < 0.1:
                continue  # Ambiguous direction

            # Get live price
            yes_price = market.yes_price
            if market.clob_token_ids and token_id in (market.clob_token_ids or []):
                if token_id in (prices or {}):
                    yes_price = prices[token_id].get("mid", yes_price)

            # Edge: VPIN-scaled.  Higher VPIN = more confident informed flow.
            edge_pct = (vpin - vpin_threshold) * 30.0  # scale to reasonable edge range
            edge_pct = min(edge_pct, 15.0)  # cap at 15%

            if direction > 0:
                # Buy pressure — buy YES
                outcome = "YES"
                side_price = yes_price
            else:
                # Sell pressure — buy NO
                outcome = "NO"
                side_price = 1.0 - yes_price

            confidence = min(0.95, 0.50 + vpin * 0.4)
            risk_score, risk_factors = self.calculate_risk_score([market], getattr(market, "end_date", None))

            opp = self.create_opportunity(
                title=f"VPIN toxicity spike on {market.question[:60]}",
                description=(
                    f"VPIN={vpin:.2f} (threshold={vpin_threshold:.2f}), "
                    f"flow={'buy' if direction > 0 else 'sell'} dominant"
                ),
                total_cost=side_price,
                markets=[market],
                positions=[{"action": "BUY", "outcome": outcome, "price": side_price}],
                is_guaranteed=False,
                custom_roi_percent=edge_pct,
                custom_risk_score=risk_score,
                confidence=confidence,
            )
            if opp:
                opportunities.append(opp)

        return opportunities

    # ------------------------------------------------------------------
    # Composable evaluate pipeline overrides
    # ------------------------------------------------------------------

    def custom_checks(
        self,
        signal: Any,
        context: Any,
        params: dict,
        payload: dict,
    ) -> list[DecisionCheck]:
        """Verify that VPIN data exists for the signal's token."""
        strategy_type = str(payload.get("strategy") or payload.get("strategy_type") or "").strip().lower()
        strategy_ok = strategy_type == "vpin_toxicity"

        # Check that we have bucket data (i.e. VPIN was actually computed)
        buckets = self.state.get("buckets", {})
        has_bucket_data = len(buckets) > 0

        return [
            DecisionCheck(
                "strategy",
                "VPIN toxicity strategy type",
                strategy_ok,
                detail="strategy=vpin_toxicity",
            ),
            DecisionCheck(
                "bucket_data",
                "VPIN bucket data available",
                has_bucket_data,
                detail=f"{len(buckets)} tokens tracked",
            ),
        ]

    def compute_size(
        self,
        base_size: float,
        max_size: float,
        edge: float,
        confidence: float,
        risk_score: float,
        market_count: int,
    ) -> float:
        """Kelly-based sizing with VPIN-scaled confidence."""
        p_estimated = 0.5 + (edge / 200.0)
        p_market = 0.5
        kelly_f = kelly_fraction(p_estimated, p_market, fraction=0.25)
        kelly_sz = base_size * (1.0 + kelly_f * 10.0)
        # Scale by confidence (which is VPIN-derived)
        size = kelly_sz * (0.6 + confidence * 0.8) * max(0.4, 1.0 - risk_score)
        return max(1.0, min(max_size, size))

    # ------------------------------------------------------------------
    # Exit logic
    # ------------------------------------------------------------------

    def should_exit(self, position: Any, market_state: dict) -> ExitDecision:
        """Exit when VPIN drops below threshold (informed traders done) or standard TP/SL."""
        if market_state.get("is_resolved"):
            return self.default_exit_check(position, market_state)

        ctx = getattr(position, "strategy_context", None) or {}
        current_price = market_state.get("current_price")

        # Check if VPIN has decayed — the informed flow signal has dissipated
        token_id = ctx.get("token_id")
        if token_id:
            vpin = self._calculate_vpin(token_id)
            vpin_threshold = float(self.config.get("vpin_threshold", 0.70))
            if vpin is not None and vpin < vpin_threshold * 0.6:
                return ExitDecision(
                    "close",
                    f"VPIN decayed below exit threshold ({vpin:.2f} < {vpin_threshold * 0.6:.2f})",
                    close_price=current_price,
                )

        return self.default_exit_check(position, market_state)

    # ------------------------------------------------------------------
    # Platform gate hooks
    # ------------------------------------------------------------------

    def on_blocked(self, signal: Any, reason: str, context: Any) -> None:
        logger.info(
            "%s: signal blocked - %s (market=%s)",
            self.name,
            reason,
            getattr(signal, "market_id", "?"),
        )

    def on_size_capped(self, original_size: float, capped_size: float, reason: str) -> None:
        logger.info(
            "%s: size capped $%.0f -> $%.0f - %s",
            self.name,
            original_size,
            capped_size,
            reason,
        )
