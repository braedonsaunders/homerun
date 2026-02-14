from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RiskCheck:
    key: str
    passed: bool
    detail: str
    score: float | None = None


@dataclass
class RiskResult:
    allowed: bool
    reason: str
    checks: list[RiskCheck] = field(default_factory=list)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def evaluate_risk(
    *,
    size_usd: float,
    gross_exposure_usd: float,
    trader_open_positions: int,
    market_exposure_usd: float,
    global_limits: dict[str, Any] | None,
    trader_limits: dict[str, Any] | None,
    global_daily_realized_pnl_usd: float = 0.0,
    trader_daily_realized_pnl_usd: float = 0.0,
    trader_consecutive_losses: int = 0,
    cycle_orders_placed: int = 0,
    cooldown_active: bool = False,
) -> RiskResult:
    global_limits = global_limits or {}
    trader_limits = trader_limits or {}

    checks: list[RiskCheck] = []

    global_max_daily_loss = abs(_safe_float(global_limits.get("max_daily_loss_usd"), 500.0))
    checks.append(
        RiskCheck(
            key="global_daily_loss",
            passed=global_daily_realized_pnl_usd > -global_max_daily_loss,
            detail=f"realized={global_daily_realized_pnl_usd:.2f} floor={-global_max_daily_loss:.2f}",
            score=global_daily_realized_pnl_usd,
        )
    )

    trader_max_daily_loss = abs(
        _safe_float(
            trader_limits.get("max_daily_loss_usd"),
            global_max_daily_loss,
        )
    )
    checks.append(
        RiskCheck(
            key="trader_daily_loss",
            passed=trader_daily_realized_pnl_usd > -trader_max_daily_loss,
            detail=f"realized={trader_daily_realized_pnl_usd:.2f} floor={-trader_max_daily_loss:.2f}",
            score=trader_daily_realized_pnl_usd,
        )
    )

    halt_on_losses = _safe_bool(trader_limits.get("halt_on_consecutive_losses"), False)
    max_consecutive_losses = max(1, _safe_int(trader_limits.get("max_consecutive_losses"), 4))
    checks.append(
        RiskCheck(
            key="trader_loss_streak",
            passed=(not halt_on_losses) or trader_consecutive_losses < max_consecutive_losses,
            detail=(
                f"streak={trader_consecutive_losses} max={max_consecutive_losses}"
                if halt_on_losses
                else "disabled"
            ),
            score=float(trader_consecutive_losses),
        )
    )

    checks.append(
        RiskCheck(
            key="trader_cooldown",
            passed=not bool(cooldown_active),
            detail="cooldown active" if cooldown_active else "cooldown clear",
            score=1.0 if cooldown_active else 0.0,
        )
    )

    max_orders_per_cycle = max(1, _safe_int(trader_limits.get("max_orders_per_cycle"), 50))
    checks.append(
        RiskCheck(
            key="trader_orders_per_cycle",
            passed=(cycle_orders_placed + 1) <= max_orders_per_cycle,
            detail=f"next={cycle_orders_placed + 1} max={max_orders_per_cycle}",
            score=float(cycle_orders_placed + 1),
        )
    )

    max_trade_notional = max(1.0, _safe_float(trader_limits.get("max_trade_notional_usd"), 1_000_000.0))
    checks.append(
        RiskCheck(
            key="trader_trade_notional",
            passed=max(0.0, size_usd) <= max_trade_notional,
            detail=f"size={max(0.0, size_usd):.2f} max={max_trade_notional:.2f}",
            score=max(0.0, size_usd),
        )
    )

    max_gross = _safe_float(global_limits.get("max_gross_exposure_usd"), 5000.0)
    next_gross = max(0.0, gross_exposure_usd) + max(0.0, size_usd)
    checks.append(
        RiskCheck(
            key="global_gross_exposure",
            passed=next_gross <= max_gross,
            detail=f"next={next_gross:.2f} max={max_gross:.2f}",
            score=next_gross,
        )
    )

    max_trader_orders = _safe_int(
        trader_limits.get("max_open_positions", trader_limits.get("max_open_orders")),
        10,
    )
    checks.append(
        RiskCheck(
            key="trader_open_positions",
            passed=(trader_open_positions + 1) <= max_trader_orders,
            detail=f"next={trader_open_positions + 1} max={max_trader_orders}",
            score=float(trader_open_positions + 1),
        )
    )

    max_per_market = _safe_float(trader_limits.get("max_per_market_exposure_usd"), 500.0)
    next_market = max(0.0, market_exposure_usd) + max(0.0, size_usd)
    checks.append(
        RiskCheck(
            key="trader_market_exposure",
            passed=next_market <= max_per_market,
            detail=f"next={next_market:.2f} max={max_per_market:.2f}",
            score=next_market,
        )
    )

    failed = [check for check in checks if not check.passed]
    if failed:
        return RiskResult(
            allowed=False,
            reason=f"Risk blocked: {failed[0].key}",
            checks=checks,
        )

    return RiskResult(
        allowed=True,
        reason="Risk checks passed",
        checks=checks,
    )
