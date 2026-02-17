from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from utils.converters import safe_float, safe_int

# ---------------------------------------------------------------------------
# ABSOLUTE SAFETY CEILINGS FOR LIVE TRADING
#
# These hard-coded constants are non-configurable safety limits that cap the
# user-supplied values when the system is operating in live mode.  No
# database configuration, environment variable, or operator action can
# exceed these ceilings.  They exist to prevent catastrophic loss from
# misconfiguration (e.g. setting max_gross_exposure_usd to 999999999).
#
# To change these values a code deploy is required -- that is by design.
# ---------------------------------------------------------------------------
ABSOLUTE_MAX_GROSS_EXPOSURE_USD: float = 50_000.0
ABSOLUTE_MAX_DAILY_LOSS_USD: float = 10_000.0
ABSOLUTE_MAX_TRADE_NOTIONAL_USD: float = 10_000.0
ABSOLUTE_MAX_PER_MARKET_EXPOSURE_USD: float = 25_000.0


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
    global_unrealized_pnl_usd: float = 0.0,
    trader_unrealized_pnl_usd: float = 0.0,
    trader_consecutive_losses: int = 0,
    cycle_orders_placed: int = 0,
    cooldown_active: bool = False,
    mode: str = "paper",
) -> RiskResult:
    global_limits = global_limits or {}
    trader_limits = trader_limits or {}
    is_live = str(mode).strip().lower() == "live"

    checks: list[RiskCheck] = []

    # --- Daily loss limits ---
    global_max_daily_loss = abs(safe_float(global_limits.get("max_daily_loss_usd"), 500.0))
    # Safety ceiling: in live mode, cap at the hard-coded absolute maximum.
    if is_live:
        global_max_daily_loss = min(global_max_daily_loss, ABSOLUTE_MAX_DAILY_LOSS_USD)
    checks.append(
        RiskCheck(
            key="global_daily_loss",
            passed=global_daily_realized_pnl_usd > -global_max_daily_loss,
            detail=f"realized={global_daily_realized_pnl_usd:.2f} floor={-global_max_daily_loss:.2f}",
            score=global_daily_realized_pnl_usd,
        )
    )

    trader_max_daily_loss = abs(
        safe_float(
            trader_limits.get("max_daily_loss_usd"),
            global_max_daily_loss,
        )
    )
    # Safety ceiling: in live mode, cap at the hard-coded absolute maximum.
    if is_live:
        trader_max_daily_loss = min(trader_max_daily_loss, ABSOLUTE_MAX_DAILY_LOSS_USD)
    checks.append(
        RiskCheck(
            key="trader_daily_loss",
            passed=trader_daily_realized_pnl_usd > -trader_max_daily_loss,
            detail=f"realized={trader_daily_realized_pnl_usd:.2f} floor={-trader_max_daily_loss:.2f}",
            score=trader_daily_realized_pnl_usd,
        )
    )

    # --- Total daily loss (realized + unrealized mark-to-market) ---
    # Open positions that are deeply underwater should also trigger the
    # daily loss limit, not just realized losses.
    global_total_daily_pnl = global_daily_realized_pnl_usd + global_unrealized_pnl_usd
    checks.append(
        RiskCheck(
            key="global_daily_total_loss",
            passed=global_total_daily_pnl > -global_max_daily_loss,
            detail=(
                f"realized={global_daily_realized_pnl_usd:.2f} "
                f"unrealized={global_unrealized_pnl_usd:.2f} "
                f"total={global_total_daily_pnl:.2f} "
                f"floor={-global_max_daily_loss:.2f}"
            ),
            score=global_total_daily_pnl,
        )
    )

    trader_total_daily_pnl = trader_daily_realized_pnl_usd + trader_unrealized_pnl_usd
    checks.append(
        RiskCheck(
            key="trader_daily_total_loss",
            passed=trader_total_daily_pnl > -trader_max_daily_loss,
            detail=(
                f"realized={trader_daily_realized_pnl_usd:.2f} "
                f"unrealized={trader_unrealized_pnl_usd:.2f} "
                f"total={trader_total_daily_pnl:.2f} "
                f"floor={-trader_max_daily_loss:.2f}"
            ),
            score=trader_total_daily_pnl,
        )
    )

    halt_on_losses = _safe_bool(trader_limits.get("halt_on_consecutive_losses"), False)
    max_consecutive_losses = max(1, safe_int(trader_limits.get("max_consecutive_losses"), 4))
    checks.append(
        RiskCheck(
            key="trader_loss_streak",
            passed=(not halt_on_losses) or trader_consecutive_losses < max_consecutive_losses,
            detail=(
                f"streak={trader_consecutive_losses} max={max_consecutive_losses}" if halt_on_losses else "disabled"
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

    max_orders_per_cycle = max(1, safe_int(trader_limits.get("max_orders_per_cycle"), 50))
    checks.append(
        RiskCheck(
            key="trader_orders_per_cycle",
            passed=(cycle_orders_placed + 1) <= max_orders_per_cycle,
            detail=f"next={cycle_orders_placed + 1} max={max_orders_per_cycle}",
            score=float(cycle_orders_placed + 1),
        )
    )

    # --- Trade notional limit ---
    # Derive a sensible default from max_gross_exposure (10% of gross cap, floor $50)
    # rather than the previous $1M which provided no real protection.
    _gross_cap = safe_float(global_limits.get("max_gross_exposure_usd"), 5000.0)
    _notional_default = max(50.0, _gross_cap * 0.10)
    max_trade_notional = max(1.0, safe_float(trader_limits.get("max_trade_notional_usd"), _notional_default))
    # Safety ceiling: in live mode, cap at the hard-coded absolute maximum.
    if is_live:
        max_trade_notional = min(max_trade_notional, ABSOLUTE_MAX_TRADE_NOTIONAL_USD)
    checks.append(
        RiskCheck(
            key="trader_trade_notional",
            passed=max(0.0, size_usd) <= max_trade_notional,
            detail=f"size={max(0.0, size_usd):.2f} max={max_trade_notional:.2f}",
            score=max(0.0, size_usd),
        )
    )

    # --- Gross exposure limit ---
    max_gross = safe_float(global_limits.get("max_gross_exposure_usd"), 5000.0)
    # Safety ceiling: in live mode, cap at the hard-coded absolute maximum.
    if is_live:
        max_gross = min(max_gross, ABSOLUTE_MAX_GROSS_EXPOSURE_USD)
    next_gross = max(0.0, gross_exposure_usd) + max(0.0, size_usd)
    checks.append(
        RiskCheck(
            key="global_gross_exposure",
            passed=next_gross <= max_gross,
            detail=f"next={next_gross:.2f} max={max_gross:.2f}",
            score=next_gross,
        )
    )

    max_trader_orders = safe_int(
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

    # --- Per-market exposure limit ---
    max_per_market = safe_float(trader_limits.get("max_per_market_exposure_usd"), 500.0)
    # Safety ceiling: in live mode, cap at the hard-coded absolute maximum.
    if is_live:
        max_per_market = min(max_per_market, ABSOLUTE_MAX_PER_MARKET_EXPOSURE_USD)
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
