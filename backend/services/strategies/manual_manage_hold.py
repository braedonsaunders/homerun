from __future__ import annotations

from typing import Any

from services.strategies.base import BaseStrategy, ExitDecision, StrategyDecision
from utils.converters import safe_float, safe_int


class ManualManageHoldStrategy(BaseStrategy):
    name = "Manual Manage Hold"
    description = "Manage manually adopted positions with downside protection and profit lock-in."
    source_key = "manual"
    worker_affinity = "manual"
    allow_new_entries = False
    default_config = {
        "min_hold_minutes": 2.0,
        "hard_stop_loss_pct": 18.0,
        "backside_activation_profit_pct": 3.0,
        "backside_drawdown_pct": 1.6,
        "backside_confirm_cycles": 2,
        "breakeven_arm_profit_pct": 2.2,
        "breakeven_buffer_pct": 0.1,
        "near_resolution_window_seconds": 300.0,
        "near_resolution_hold_profit_pct": 4.0,
        "near_resolution_stop_loss_pct": 3.0,
        "neutral_exit_min_age_minutes": 120.0,
        "neutral_exit_band_pct": 0.25,
        "max_hold_minutes": 0.0,
    }

    def evaluate(self, signal, context):
        return StrategyDecision(
            decision="blocked",
            reason="Manual manage strategy blocks new entries",
            checks=[],
        )

    @staticmethod
    def _cfg_float(config: dict[str, Any], key: str, default: float) -> float:
        parsed = safe_float(config.get(key), default)
        return float(default if parsed is None else parsed)

    @staticmethod
    def _cfg_int(config: dict[str, Any], key: str, default: int) -> int:
        parsed = safe_int(config.get(key), default)
        return int(parsed)

    def _effective_config(self, position: Any) -> dict[str, Any]:
        merged = dict(getattr(self, "config", {}) or {})
        raw_position_config = getattr(position, "config", None)
        if isinstance(raw_position_config, dict):
            merged.update(raw_position_config)
        return merged

    @staticmethod
    def _ensure_strategy_context(position: Any) -> dict[str, Any]:
        strategy_context = getattr(position, "strategy_context", None)
        if isinstance(strategy_context, dict):
            return strategy_context
        strategy_context = {}
        position.strategy_context = strategy_context
        return strategy_context

    @staticmethod
    def _peak_price(position: Any, strategy_context: dict[str, Any], current_price: float, entry_price: float) -> float:
        stored_peak = safe_float(strategy_context.get("_manual_peak_price"))
        observed_high = safe_float(getattr(position, "highest_price", None))
        candidates = [entry_price, current_price]
        if stored_peak is not None and stored_peak > 0.0:
            candidates.append(stored_peak)
        if observed_high is not None and observed_high > 0.0:
            candidates.append(observed_high)
        peak_price = max(candidates)
        strategy_context["_manual_peak_price"] = float(peak_price)
        return float(peak_price)

    @staticmethod
    def _update_decline_streak(strategy_context: dict[str, Any], current_price: float) -> int:
        last_price = safe_float(strategy_context.get("_manual_last_price"))
        streak = int(safe_int(strategy_context.get("_manual_decline_streak"), 0))
        epsilon = 1e-6
        if last_price is None:
            streak = 0
        elif current_price < (last_price - epsilon):
            streak += 1
        elif current_price > (last_price + epsilon):
            streak = 0
        strategy_context["_manual_last_price"] = float(current_price)
        strategy_context["_manual_decline_streak"] = int(max(0, streak))
        return int(max(0, streak))

    def should_exit(self, position: Any, market_state: dict) -> ExitDecision:
        if market_state.get("is_resolved"):
            return self.default_exit_check(position, market_state)

        current_price = safe_float(market_state.get("current_price"))
        if current_price is None or current_price < 0.0:
            return ExitDecision(action="hold", reason="Manual manage: waiting for current price")

        entry_price = safe_float(getattr(position, "entry_price", None))
        if entry_price is None or entry_price <= 0.0:
            return ExitDecision(action="hold", reason="Manual manage: waiting for entry price")

        config = self._effective_config(position)
        strategy_context = self._ensure_strategy_context(position)

        age_minutes = float(max(0.0, safe_float(getattr(position, "age_minutes", 0.0), 0.0) or 0.0))
        pnl_pct = float(((current_price - entry_price) / entry_price) * 100.0)

        peak_price = self._peak_price(position, strategy_context, current_price, entry_price)
        peak_gain_pct = ((peak_price - entry_price) / entry_price * 100.0) if entry_price > 0.0 else 0.0
        drawdown_from_peak_pct = ((peak_price - current_price) / peak_price * 100.0) if peak_price > 0.0 else 0.0
        decline_streak = self._update_decline_streak(strategy_context, float(current_price))

        min_hold_minutes = max(0.0, self._cfg_float(config, "min_hold_minutes", 2.0))
        min_hold_passed = age_minutes >= min_hold_minutes

        hard_stop_loss_pct = max(0.0, self._cfg_float(config, "hard_stop_loss_pct", 18.0))
        if min_hold_passed and hard_stop_loss_pct > 0.0 and pnl_pct <= -hard_stop_loss_pct:
            return ExitDecision(
                action="close",
                reason=f"Manual hard stop loss ({pnl_pct:.2f}% <= -{hard_stop_loss_pct:.2f}%)",
                close_price=float(current_price),
            )

        breakeven_arm_profit_pct = max(0.0, self._cfg_float(config, "breakeven_arm_profit_pct", 2.2))
        if peak_gain_pct >= breakeven_arm_profit_pct:
            strategy_context["_manual_breakeven_armed"] = True
        breakeven_armed = bool(strategy_context.get("_manual_breakeven_armed", False))

        breakeven_buffer_pct = self._cfg_float(config, "breakeven_buffer_pct", 0.1)
        if breakeven_armed and min_hold_passed and pnl_pct <= breakeven_buffer_pct:
            return ExitDecision(
                action="close",
                reason=f"Manual breakeven protect ({pnl_pct:.2f}% <= {breakeven_buffer_pct:.2f}%)",
                close_price=float(current_price),
            )

        backside_activation_profit_pct = max(0.0, self._cfg_float(config, "backside_activation_profit_pct", 3.0))
        backside_drawdown_pct = max(0.0, self._cfg_float(config, "backside_drawdown_pct", 1.6))
        backside_confirm_cycles = max(1, self._cfg_int(config, "backside_confirm_cycles", 2))
        if (
            min_hold_passed
            and peak_gain_pct >= backside_activation_profit_pct
            and drawdown_from_peak_pct >= backside_drawdown_pct
            and decline_streak >= backside_confirm_cycles
        ):
            return ExitDecision(
                action="close",
                reason=(
                    f"Manual backside peak exit (drawdown {drawdown_from_peak_pct:.2f}% "
                    f"after peak gain {peak_gain_pct:.2f}%)"
                ),
                close_price=float(current_price),
            )

        seconds_left = self._seconds_left_for_position(position, market_state)
        near_resolution_window_seconds = max(0.0, self._cfg_float(config, "near_resolution_window_seconds", 300.0))
        near_resolution_hold_profit_pct = self._cfg_float(config, "near_resolution_hold_profit_pct", 4.0)
        near_resolution_stop_loss_pct = max(0.0, self._cfg_float(config, "near_resolution_stop_loss_pct", 3.0))
        if seconds_left is not None and near_resolution_window_seconds > 0.0 and seconds_left <= near_resolution_window_seconds:
            if pnl_pct >= near_resolution_hold_profit_pct:
                return ExitDecision(
                    action="hold",
                    reason=(
                        f"Manual near-resolution hold ({seconds_left:.0f}s left, "
                        f"pnl {pnl_pct:.2f}% >= {near_resolution_hold_profit_pct:.2f}%)"
                    ),
                )
            if pnl_pct <= -near_resolution_stop_loss_pct:
                return ExitDecision(
                    action="close",
                    reason=(
                        f"Manual near-resolution risk exit ({pnl_pct:.2f}% <= "
                        f"-{near_resolution_stop_loss_pct:.2f}%)"
                    ),
                    close_price=float(current_price),
                )

        neutral_exit_min_age_minutes = max(0.0, self._cfg_float(config, "neutral_exit_min_age_minutes", 120.0))
        neutral_exit_band_pct = max(0.0, self._cfg_float(config, "neutral_exit_band_pct", 0.25))
        if age_minutes >= neutral_exit_min_age_minutes and abs(pnl_pct) <= neutral_exit_band_pct:
            return ExitDecision(
                action="close",
                reason=(
                    f"Manual neutral recycle (age {age_minutes:.0f}m, "
                    f"abs pnl {abs(pnl_pct):.2f}% <= {neutral_exit_band_pct:.2f}%)"
                ),
                close_price=float(current_price),
            )

        max_hold_minutes = max(0.0, self._cfg_float(config, "max_hold_minutes", 0.0))
        if max_hold_minutes > 0.0 and age_minutes >= max_hold_minutes and pnl_pct > 0.0:
            return ExitDecision(
                action="close",
                reason=f"Manual max hold profit lock ({age_minutes:.0f}m >= {max_hold_minutes:.0f}m)",
                close_price=float(current_price),
            )

        return ExitDecision(
            action="hold",
            reason=(
                f"Manual manage hold (pnl {pnl_pct:.2f}%, peak {peak_gain_pct:.2f}%, "
                f"drawdown {drawdown_from_peak_pct:.2f}%)"
            ),
        )
