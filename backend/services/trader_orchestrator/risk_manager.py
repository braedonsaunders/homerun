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


def evaluate_risk(
    *,
    size_usd: float,
    gross_exposure_usd: float,
    trader_open_orders: int,
    market_exposure_usd: float,
    global_limits: dict[str, Any] | None,
    trader_limits: dict[str, Any] | None,
) -> RiskResult:
    global_limits = global_limits or {}
    trader_limits = trader_limits or {}

    checks: list[RiskCheck] = []

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

    max_trader_orders = _safe_int(trader_limits.get("max_open_orders"), 10)
    checks.append(
        RiskCheck(
            key="trader_open_orders",
            passed=(trader_open_orders + 1) <= max_trader_orders,
            detail=f"next={trader_open_orders + 1} max={max_trader_orders}",
            score=float(trader_open_orders + 1),
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
