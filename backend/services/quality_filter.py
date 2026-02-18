from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from config import settings


def _make_aware(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


@dataclass
class FilterResult:
    """Result of a single quality filter check."""
    filter_name: str
    passed: bool
    reason: str
    threshold: Any = None
    actual_value: Any = None


@dataclass
class QualityReport:
    """Full audit trail of all quality filters applied to an opportunity."""
    opportunity_id: str
    passed: bool
    filters: list[FilterResult] = field(default_factory=list)

    @property
    def rejection_reasons(self) -> list[str]:
        return [f.reason for f in self.filters if not f.passed]


class QualityFilterPipeline:
    """Runs all opportunity quality filters with full audit trail.

    Each filter returns a FilterResult with:
    - filter_name: machine-readable identifier
    - passed: True/False
    - reason: human-readable explanation
    - threshold: the threshold value used
    - actual_value: the actual value that was compared
    """

    def evaluate(self, opp: Any) -> QualityReport:
        """Run all quality filters on an opportunity.

        Args:
            opp: ArbitrageOpportunity with fields: roi_percent, is_guaranteed,
                 markets (list of dicts with 'liquidity'), positions_to_take,
                 resolution_date, min_liquidity, max_position_size,
                 net_profit, total_cost, stable_id, id
        """
        filters = [
            self._check_min_roi(opp),
            self._check_directional_roi_cap(opp),
            self._check_plausible_roi(opp),
            self._check_max_legs(opp),
            self._check_leg_liquidity(opp),
            self._check_min_liquidity(opp),
            self._check_min_position_size(opp),
            self._check_min_absolute_profit(opp),
            self._check_resolution_timeframe(opp),
            self._check_annualized_roi(opp),
        ]

        passed = all(f.passed for f in filters)
        opp_id = getattr(opp, "stable_id", None) or getattr(opp, "id", "unknown")
        return QualityReport(
            opportunity_id=str(opp_id),
            passed=passed,
            filters=filters,
        )

    def _check_min_roi(self, opp: Any) -> FilterResult:
        roi = float(getattr(opp, "roi_percent", 0) or 0)
        threshold = settings.MIN_PROFIT_THRESHOLD * 100
        passed = roi >= threshold
        return FilterResult(
            filter_name="min_roi",
            passed=passed,
            reason=(f"ROI {roi:.2f}% >= {threshold:.2f}% minimum" if passed
                    else f"ROI {roi:.2f}% below {threshold:.2f}% minimum"),
            threshold=threshold,
            actual_value=roi,
        )

    def _check_directional_roi_cap(self, opp: Any) -> FilterResult:
        roi = float(getattr(opp, "roi_percent", 0) or 0)
        is_guaranteed = bool(getattr(opp, "is_guaranteed", True))
        cap = 120.0
        if is_guaranteed:
            return FilterResult("directional_roi_cap", True,
                                "Guaranteed spread (not directional)", cap, roi)
        passed = roi <= cap
        return FilterResult(
            "directional_roi_cap", passed,
            (f"Directional ROI {roi:.1f}% <= {cap:.0f}% cap" if passed
             else f"Directional ROI {roi:.1f}% exceeds {cap:.0f}% cap (likely artifact)"),
            cap, roi,
        )

    def _check_plausible_roi(self, opp: Any) -> FilterResult:
        roi = float(getattr(opp, "roi_percent", 0) or 0)
        is_guaranteed = bool(getattr(opp, "is_guaranteed", True))
        cap = float(settings.MAX_PLAUSIBLE_ROI)
        if not is_guaranteed:
            return FilterResult("plausible_roi", True,
                                "Directional strategy (plausible ROI check skipped)", cap, roi)
        passed = roi <= cap
        return FilterResult(
            "plausible_roi", passed,
            (f"Guaranteed ROI {roi:.1f}% <= {cap:.0f}% plausible cap" if passed
             else f"Guaranteed ROI {roi:.1f}% exceeds {cap:.0f}% plausible cap (likely stale data)"),
            cap, roi,
        )

    def _check_max_legs(self, opp: Any) -> FilterResult:
        markets = getattr(opp, "markets", []) or []
        num_legs = len(markets)
        cap = int(settings.MAX_TRADE_LEGS)
        passed = num_legs <= cap
        return FilterResult(
            "max_legs", passed,
            (f"{num_legs} legs <= {cap} maximum" if passed
             else f"{num_legs} legs exceeds {cap} maximum (slippage compounds per leg)"),
            cap, num_legs,
        )

    def _check_leg_liquidity(self, opp: Any) -> FilterResult:
        markets = getattr(opp, "markets", []) or []
        num_legs = len(markets)
        if num_legs <= 1:
            return FilterResult("leg_liquidity", True,
                                "Single-leg trade (leg liquidity check skipped)", 0, 0)
        min_per_leg = float(getattr(settings, "MIN_LIQUIDITY_PER_LEG", 500.0))
        required = min_per_leg * num_legs
        total_liquidity = sum(
            float((m.get("liquidity") if isinstance(m, dict) else getattr(m, "liquidity", 0)) or 0)
            for m in markets
        )
        passed = total_liquidity >= required
        return FilterResult(
            "leg_liquidity", passed,
            (f"Total liquidity ${total_liquidity:,.0f} >= ${required:,.0f} required ({num_legs} legs)" if passed
             else f"Total liquidity ${total_liquidity:,.0f} below ${required:,.0f} required for {num_legs} legs"),
            required, total_liquidity,
        )

    def _check_min_liquidity(self, opp: Any) -> FilterResult:
        min_liq = float(getattr(opp, "min_liquidity", 0) or 0)
        threshold = float(settings.MIN_LIQUIDITY_HARD)
        passed = min_liq >= threshold
        return FilterResult(
            "min_liquidity", passed,
            (f"Min liquidity ${min_liq:,.0f} >= ${threshold:,.0f} floor" if passed
             else f"Min liquidity ${min_liq:,.0f} below ${threshold:,.0f} floor"),
            threshold, min_liq,
        )

    def _check_min_position_size(self, opp: Any) -> FilterResult:
        max_pos = float(getattr(opp, "max_position_size", 0) or 0)
        threshold = float(settings.MIN_POSITION_SIZE)
        passed = max_pos >= threshold
        return FilterResult(
            "min_position_size", passed,
            (f"Max position ${max_pos:,.0f} >= ${threshold:,.0f} minimum" if passed
             else f"Max position ${max_pos:,.0f} below ${threshold:,.0f} minimum (market too thin)"),
            threshold, max_pos,
        )

    def _check_min_absolute_profit(self, opp: Any) -> FilterResult:
        max_pos = float(getattr(opp, "max_position_size", 0) or 0)
        net = float(getattr(opp, "net_profit", 0) or 0)
        cost = float(getattr(opp, "total_cost", 0) or 0)
        absolute = max_pos * (net / cost) if cost > 0 else 0
        threshold = float(settings.MIN_ABSOLUTE_PROFIT)
        passed = absolute >= threshold
        return FilterResult(
            "min_absolute_profit", passed,
            (f"Absolute profit ${absolute:,.2f} >= ${threshold:,.2f} minimum" if passed
             else f"Absolute profit ${absolute:,.2f} below ${threshold:,.2f} minimum"),
            threshold, absolute,
        )

    def _check_resolution_timeframe(self, opp: Any) -> FilterResult:
        resolution_date = getattr(opp, "resolution_date", None)
        if resolution_date is None:
            return FilterResult("resolution_timeframe", True,
                                "No resolution date (timeframe check skipped)", None, None)
        resolution_aware = _make_aware(resolution_date)
        now = datetime.now(timezone.utc)
        days_until = (resolution_aware - now).total_seconds() / 86400.0
        max_days = float(settings.MAX_RESOLUTION_MONTHS * 30)
        passed = days_until <= max_days
        return FilterResult(
            "resolution_timeframe", passed,
            (f"{days_until:.0f} days to resolution <= {max_days:.0f} day maximum" if passed
             else f"{days_until:.0f} days to resolution exceeds {max_days:.0f} day maximum"),
            max_days, days_until,
        )

    def _check_annualized_roi(self, opp: Any) -> FilterResult:
        resolution_date = getattr(opp, "resolution_date", None)
        if resolution_date is None:
            return FilterResult("annualized_roi", True,
                                "No resolution date (annualized ROI check skipped)", None, None)
        roi = float(getattr(opp, "roi_percent", 0) or 0)
        resolution_aware = _make_aware(resolution_date)
        now = datetime.now(timezone.utc)
        days_until = max((resolution_aware - now).total_seconds() / 86400.0, 1.0)
        annualized = roi * (365.0 / days_until)
        threshold = float(settings.MIN_ANNUALIZED_ROI)
        passed = annualized >= threshold
        return FilterResult(
            "annualized_roi", passed,
            (f"Annualized ROI {annualized:.1f}% >= {threshold:.1f}% minimum" if passed
             else f"Annualized ROI {annualized:.1f}% below {threshold:.1f}% minimum"),
            threshold, annualized,
        )


quality_filter = QualityFilterPipeline()
