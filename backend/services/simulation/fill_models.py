from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from services.optimization.execution_estimator import ExecutionEstimator, ExecutionEstimatorConfig


@dataclass
class FillConfig:
    slippage_bps: float = 5.0
    fee_bps: float = 200.0  # 2%
    latency_ms: float = 350.0
    displayed_depth_factor: float = 0.88
    time_in_force_seconds: float = 6.0


class FillModel:
    """Execution fill models for replay and shadow simulation."""

    _execution_estimator = ExecutionEstimator()

    @staticmethod
    def _as_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return default

    @staticmethod
    def _is_buy_side(direction: str) -> bool:
        side = str(direction or "").strip().lower()
        return side in {"buy", "buy_yes", "buy_no", "yes", "no", "long"}

    @classmethod
    def intrabar_touch_fill(
        cls,
        *,
        direction: str,
        target_price: float,
        notional_usd: float,
        candle: dict[str, Any],
        config: FillConfig,
    ) -> dict[str, Any]:
        """Fill if target price is touched inside candle high/low range."""
        low = cls._as_float(candle.get("low"), cls._as_float(candle.get("l"), 0.0))
        high = cls._as_float(candle.get("high"), cls._as_float(candle.get("h"), 0.0))
        ts_ms = int(cls._as_float(candle.get("t"), 0.0))
        raw_target = max(0.0001, min(0.9999, cls._as_float(target_price, 0.0)))
        notional = max(0.0, cls._as_float(notional_usd, 0.0))

        touched = low <= raw_target <= high if high >= low else False
        if not touched or notional <= 0:
            return {
                "filled": False,
                "fill_price": None,
                "quantity": 0.0,
                "fees_usd": 0.0,
                "slippage_bps": float(config.slippage_bps),
                "event_ts_ms": ts_ms,
            }

        candle_volume = max(0.0, cls._as_float(candle.get("volume"), cls._as_float(candle.get("v"), 0.0)))
        if candle_volume > 0:
            volume_notional = candle_volume * raw_target
            participation_cap = volume_notional * 0.20
            fill_ratio = min(1.0, participation_cap / notional) if notional > 0 else 0.0
        else:
            fill_ratio = 0.35

        if fill_ratio <= 0:
            return {
                "filled": False,
                "fill_price": None,
                "quantity": 0.0,
                "fees_usd": 0.0,
                "slippage_bps": float(config.slippage_bps),
                "event_ts_ms": ts_ms,
                "fill_ratio": 0.0,
                "reason": "bar_touched_without_trade_volume",
            }

        slippage = abs(float(config.slippage_bps or 0.0)) / 10000.0
        is_buy = cls._is_buy_side(direction)
        fill_price = raw_target * (1.0 + slippage if is_buy else 1.0 - slippage)
        fill_price = max(0.0001, min(0.9999, fill_price))

        filled_notional = notional * fill_ratio
        quantity = filled_notional / fill_price if fill_price > 0 else 0.0
        fees = filled_notional * (abs(float(config.fee_bps or 0.0)) / 10000.0)

        return {
            "filled": True,
            "fill_price": fill_price,
            "quantity": quantity,
            "filled_notional_usd": filled_notional,
            "fees_usd": fees,
            "slippage_bps": float(config.slippage_bps),
            "event_ts_ms": ts_ms,
            "fill_ratio": fill_ratio,
            "reason": "intrabar_touch_with_volume_participation",
        }

    @classmethod
    def order_book_fill(
        cls,
        *,
        side: str,
        notional_usd: float,
        order_book: Any,
        limit_price: float | None,
        order_type: str,
        recent_trades: list[Any] | None,
        book_age_ms: float | None,
        config: FillConfig,
    ) -> dict[str, Any]:
        estimate = cls._execution_estimator.estimate_order(
            order_book=order_book,
            side=side,
            size_usd=max(0.0, cls._as_float(notional_usd, 0.0)),
            limit_price=limit_price,
            order_type=order_type,
            recent_trades=recent_trades or [],
            book_age_ms=book_age_ms,
            config=ExecutionEstimatorConfig(
                fee_bps=max(0.0, cls._as_float(config.fee_bps, 0.0)),
                latency_ms=max(0.0, cls._as_float(config.latency_ms, 0.0)),
                time_in_force_seconds=max(0.0, cls._as_float(config.time_in_force_seconds, 0.0)),
                displayed_depth_factor=max(0.0, min(1.0, cls._as_float(config.displayed_depth_factor, 0.88))),
            ),
        )
        return {
            "filled": estimate.filled_shares > 0,
            "fill_price": estimate.average_price,
            "quantity": estimate.filled_shares,
            "filled_notional_usd": estimate.filled_notional_usd,
            "fees_usd": estimate.fees_usd,
            "slippage_bps": estimate.slippage_bps,
            "fill_ratio": estimate.fill_ratio,
            "execution_estimate": estimate.to_dict(),
        }

    @classmethod
    def mark_to_market_close(
        cls,
        *,
        direction: str,
        entry_price: float,
        quantity: float,
        last_price: float,
        config: FillConfig,
    ) -> dict[str, Any]:
        """Close unresolved positions using final mark-to-market price."""
        qty = max(0.0, cls._as_float(quantity, 0.0))
        entry = max(0.0001, min(0.9999, cls._as_float(entry_price, 0.0)))
        mark = max(0.0001, min(0.9999, cls._as_float(last_price, entry)))
        notional = qty * entry
        fees = notional * (abs(float(config.fee_bps or 0.0)) / 10000.0)

        is_buy_yes = str(direction or "").strip().lower() in {"buy_yes", "yes", "long"}
        if is_buy_yes:
            pnl = (mark - entry) * qty - fees
        else:
            pnl = ((1.0 - mark) - (1.0 - entry)) * qty - fees

        return {
            "entry_price": entry,
            "exit_price": mark,
            "quantity": qty,
            "fees_usd": fees,
            "realized_pnl_usd": pnl,
        }
