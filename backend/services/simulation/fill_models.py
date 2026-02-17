from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class FillConfig:
    slippage_bps: float = 5.0
    fee_bps: float = 200.0  # 2%


class FillModel:
    """Bar-based execution fill models for execution simulation."""

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

        slippage = abs(float(config.slippage_bps or 0.0)) / 10000.0
        is_buy = cls._is_buy_side(direction)
        fill_price = raw_target * (1.0 + slippage if is_buy else 1.0 - slippage)
        fill_price = max(0.0001, min(0.9999, fill_price))

        quantity = notional / fill_price if fill_price > 0 else 0.0
        fees = notional * (abs(float(config.fee_bps or 0.0)) / 10000.0)

        return {
            "filled": True,
            "fill_price": fill_price,
            "quantity": quantity,
            "fees_usd": fees,
            "slippage_bps": float(config.slippage_bps),
            "event_ts_ms": ts_ms,
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
