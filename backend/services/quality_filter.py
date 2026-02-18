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


@dataclass
class QualityFilterOverrides:
    """Per-strategy overrides for the QualityFilterPipeline thresholds.

    Strategies set this as a class attribute on their ``BaseStrategy`` subclass::

        from services.quality_filter import QualityFilterOverrides

        class MyStrategy(BaseStrategy):
            quality_filter_overrides = QualityFilterOverrides(
                min_roi=2.0,            # Accept lower ROI than the global default
                max_roi_cap=250.0,      # Relax cap for high-edge directional signals
            )

    Any field left as ``None`` falls back to the global platform default from
    ``config.settings``. Only set what you explicitly need to override.
    """

    # Minimum ROI threshold (percent, e.g. 2.0 means 2%). Overrides
    # ``settings.MIN_PROFIT_THRESHOLD * 100``.
    min_roi: Optional[float] = None

    # Maximum ROI cap for directional (non-guaranteed) opportunities (percent).
    # Overrides the hardcoded 120% cap that flags likely pricing artifacts.
    max_roi_cap: Optional[float] = None

    # Maximum plausible ROI for guaranteed-spread opportunities (percent).
    # Overrides ``settings.MAX_PLAUSIBLE_ROI``.
    max_plausible_roi: Optional[float] = None

    # Maximum number of trade legs. Overrides ``settings.MAX_TRADE_LEGS``.
    max_legs: Optional[int] = None

    # Minimum liquidity required per leg (USD). Overrides
    # ``settings.MIN_LIQUIDITY_PER_LEG`` (default 500.0).
    min_leg_liquidity: Optional[float] = None

    # Hard liquidity floor across all legs (USD). Overrides
    # ``settings.MIN_LIQUIDITY_HARD``.
    min_liquidity: Optional[float] = None

    # Minimum viable position size (USD). Overrides ``settings.MIN_POSITION_SIZE``.
    min_position_size: Optional[float] = None

    # Minimum absolute dollar profit at max position size (USD). Overrides
    # ``settings.MIN_ABSOLUTE_PROFIT``.
    min_absolute_profit: Optional[float] = None

    # Maximum months to resolution. Overrides ``settings.MAX_RESOLUTION_MONTHS``.
    max_resolution_months: Optional[float] = None

    # Minimum annualized ROI (percent). Overrides ``settings.MIN_ANNUALIZED_ROI``.
    min_annualized_roi: Optional[float] = None


class QualityFilterPipeline:
    """Runs all opportunity quality filters with full audit trail.

    Each filter returns a FilterResult with:
    - filter_name: machine-readable identifier
    - passed: True/False
    - reason: human-readable explanation
    - threshold: the threshold value used
    - actual_value: the actual value that was compared

    Per-strategy overrides are supported via QualityFilterOverrides. Pass an
    instance as ``overrides`` to ``evaluate()`` to relax or tighten any
    individual threshold for a specific strategy's opportunities.
    """

    def evaluate(
        self,
        opp: Any,
        overrides: Optional[QualityFilterOverrides] = None,
    ) -> QualityReport:
        """Run all quality filters on an opportunity.

        Args:
            opp: ArbitrageOpportunity with fields: roi_percent, is_guaranteed,
                 markets (list of dicts with 'liquidity'), positions_to_take,
                 resolution_date, min_liquidity, max_position_size,
                 net_profit, total_cost, stable_id, id
            overrides: Optional per-strategy threshold overrides. Any field
                 that is None falls back to the global platform default.
        """
        ov = overrides
        filters = [
            self._check_min_roi(opp, ov),
            self._check_directional_roi_cap(opp, ov),
            self._check_plausible_roi(opp, ov),
            self._check_max_legs(opp, ov),
            self._check_leg_liquidity(opp, ov),
            self._check_min_liquidity(opp, ov),
            self._check_min_position_size(opp, ov),
            self._check_min_absolute_profit(opp, ov),
            self._check_resolution_timeframe(opp, ov),
            self._check_annualized_roi(opp, ov),
        ]

        passed = all(f.passed for f in filters)
        opp_id = getattr(opp, "stable_id", None) or getattr(opp, "id", "unknown")
        return QualityReport(
            opportunity_id=str(opp_id),
            passed=passed,
            filters=filters,
        )

    def _check_min_roi(
        self,
        opp: Any,
        ov: Optional[QualityFilterOverrides] = None,
    ) -> FilterResult:
        roi = float(getattr(opp, "roi_percent", 0) or 0)
        threshold = (
            float(ov.min_roi) if ov is not None and ov.min_roi is not None else settings.MIN_PROFIT_THRESHOLD * 100
        )
        passed = roi >= threshold
        return FilterResult(
            filter_name="min_roi",
            passed=passed,
            reason=(
                f"ROI {roi:.2f}% >= {threshold:.2f}% minimum"
                if passed
                else f"ROI {roi:.2f}% below {threshold:.2f}% minimum"
            ),
            threshold=threshold,
            actual_value=roi,
        )

    def _check_directional_roi_cap(
        self,
        opp: Any,
        ov: Optional[QualityFilterOverrides] = None,
    ) -> FilterResult:
        roi = float(getattr(opp, "roi_percent", 0) or 0)
        is_guaranteed = bool(getattr(opp, "is_guaranteed", True))
        cap = float(ov.max_roi_cap) if ov is not None and ov.max_roi_cap is not None else 120.0
        if is_guaranteed:
            return FilterResult("directional_roi_cap", True, "Guaranteed spread (not directional)", cap, roi)
        passed = roi <= cap
        return FilterResult(
            "directional_roi_cap",
            passed,
            (
                f"Directional ROI {roi:.1f}% <= {cap:.0f}% cap"
                if passed
                else f"Directional ROI {roi:.1f}% exceeds {cap:.0f}% cap (likely artifact)"
            ),
            cap,
            roi,
        )

    def _check_plausible_roi(
        self,
        opp: Any,
        ov: Optional[QualityFilterOverrides] = None,
    ) -> FilterResult:
        roi = float(getattr(opp, "roi_percent", 0) or 0)
        is_guaranteed = bool(getattr(opp, "is_guaranteed", True))
        cap = (
            float(ov.max_plausible_roi)
            if ov is not None and ov.max_plausible_roi is not None
            else float(settings.MAX_PLAUSIBLE_ROI)
        )
        if not is_guaranteed:
            return FilterResult("plausible_roi", True, "Directional strategy (plausible ROI check skipped)", cap, roi)
        passed = roi <= cap
        return FilterResult(
            "plausible_roi",
            passed,
            (
                f"Guaranteed ROI {roi:.1f}% <= {cap:.0f}% plausible cap"
                if passed
                else f"Guaranteed ROI {roi:.1f}% exceeds {cap:.0f}% plausible cap (likely stale data)"
            ),
            cap,
            roi,
        )

    def _check_max_legs(
        self,
        opp: Any,
        ov: Optional[QualityFilterOverrides] = None,
    ) -> FilterResult:
        markets = getattr(opp, "markets", []) or []
        num_legs = len(markets)
        cap = int(ov.max_legs) if ov is not None and ov.max_legs is not None else int(settings.MAX_TRADE_LEGS)
        passed = num_legs <= cap
        return FilterResult(
            "max_legs",
            passed,
            (
                f"{num_legs} legs <= {cap} maximum"
                if passed
                else f"{num_legs} legs exceeds {cap} maximum (slippage compounds per leg)"
            ),
            cap,
            num_legs,
        )

    def _check_leg_liquidity(
        self,
        opp: Any,
        ov: Optional[QualityFilterOverrides] = None,
    ) -> FilterResult:
        markets = getattr(opp, "markets", []) or []
        num_legs = len(markets)
        if num_legs <= 1:
            return FilterResult("leg_liquidity", True, "Single-leg trade (leg liquidity check skipped)", 0, 0)
        min_per_leg = (
            float(ov.min_leg_liquidity)
            if ov is not None and ov.min_leg_liquidity is not None
            else float(getattr(settings, "MIN_LIQUIDITY_PER_LEG", 500.0))
        )
        required = min_per_leg * num_legs
        total_liquidity = sum(
            float((m.get("liquidity") if isinstance(m, dict) else getattr(m, "liquidity", 0)) or 0) for m in markets
        )
        passed = total_liquidity >= required
        return FilterResult(
            "leg_liquidity",
            passed,
            (
                f"Total liquidity ${total_liquidity:,.0f} >= ${required:,.0f} required ({num_legs} legs)"
                if passed
                else f"Total liquidity ${total_liquidity:,.0f} below ${required:,.0f} required for {num_legs} legs"
            ),
            required,
            total_liquidity,
        )

    def _check_min_liquidity(
        self,
        opp: Any,
        ov: Optional[QualityFilterOverrides] = None,
    ) -> FilterResult:
        min_liq = float(getattr(opp, "min_liquidity", 0) or 0)
        threshold = (
            float(ov.min_liquidity)
            if ov is not None and ov.min_liquidity is not None
            else float(settings.MIN_LIQUIDITY_HARD)
        )
        passed = min_liq >= threshold
        return FilterResult(
            "min_liquidity",
            passed,
            (
                f"Min liquidity ${min_liq:,.0f} >= ${threshold:,.0f} floor"
                if passed
                else f"Min liquidity ${min_liq:,.0f} below ${threshold:,.0f} floor"
            ),
            threshold,
            min_liq,
        )

    def _check_min_position_size(
        self,
        opp: Any,
        ov: Optional[QualityFilterOverrides] = None,
    ) -> FilterResult:
        max_pos = float(getattr(opp, "max_position_size", 0) or 0)
        threshold = (
            float(ov.min_position_size)
            if ov is not None and ov.min_position_size is not None
            else float(settings.MIN_POSITION_SIZE)
        )
        passed = max_pos >= threshold
        return FilterResult(
            "min_position_size",
            passed,
            (
                f"Max position ${max_pos:,.0f} >= ${threshold:,.0f} minimum"
                if passed
                else f"Max position ${max_pos:,.0f} below ${threshold:,.0f} minimum (market too thin)"
            ),
            threshold,
            max_pos,
        )

    def _check_min_absolute_profit(
        self,
        opp: Any,
        ov: Optional[QualityFilterOverrides] = None,
    ) -> FilterResult:
        max_pos = float(getattr(opp, "max_position_size", 0) or 0)
        net = float(getattr(opp, "net_profit", 0) or 0)
        cost = float(getattr(opp, "total_cost", 0) or 0)
        absolute = max_pos * (net / cost) if cost > 0 else 0
        threshold = (
            float(ov.min_absolute_profit)
            if ov is not None and ov.min_absolute_profit is not None
            else float(settings.MIN_ABSOLUTE_PROFIT)
        )
        passed = absolute >= threshold
        return FilterResult(
            "min_absolute_profit",
            passed,
            (
                f"Absolute profit ${absolute:,.2f} >= ${threshold:,.2f} minimum"
                if passed
                else f"Absolute profit ${absolute:,.2f} below ${threshold:,.2f} minimum"
            ),
            threshold,
            absolute,
        )

    def _check_resolution_timeframe(
        self,
        opp: Any,
        ov: Optional[QualityFilterOverrides] = None,
    ) -> FilterResult:
        resolution_date = getattr(opp, "resolution_date", None)
        if resolution_date is None:
            return FilterResult(
                "resolution_timeframe", True, "No resolution date (timeframe check skipped)", None, None
            )
        resolution_aware = _make_aware(resolution_date)
        now = datetime.now(timezone.utc)
        days_until = (resolution_aware - now).total_seconds() / 86400.0
        max_months = (
            float(ov.max_resolution_months)
            if ov is not None and ov.max_resolution_months is not None
            else float(settings.MAX_RESOLUTION_MONTHS)
        )
        max_days = max_months * 30
        passed = days_until <= max_days
        return FilterResult(
            "resolution_timeframe",
            passed,
            (
                f"{days_until:.0f} days to resolution <= {max_days:.0f} day maximum"
                if passed
                else f"{days_until:.0f} days to resolution exceeds {max_days:.0f} day maximum"
            ),
            max_days,
            days_until,
        )

    def _check_annualized_roi(
        self,
        opp: Any,
        ov: Optional[QualityFilterOverrides] = None,
    ) -> FilterResult:
        resolution_date = getattr(opp, "resolution_date", None)
        if resolution_date is None:
            return FilterResult("annualized_roi", True, "No resolution date (annualized ROI check skipped)", None, None)
        roi = float(getattr(opp, "roi_percent", 0) or 0)
        resolution_aware = _make_aware(resolution_date)
        now = datetime.now(timezone.utc)
        days_until = max((resolution_aware - now).total_seconds() / 86400.0, 1.0)
        annualized = roi * (365.0 / days_until)
        threshold = (
            float(ov.min_annualized_roi)
            if ov is not None and ov.min_annualized_roi is not None
            else float(settings.MIN_ANNUALIZED_ROI)
        )
        passed = annualized >= threshold
        return FilterResult(
            "annualized_roi",
            passed,
            (
                f"Annualized ROI {annualized:.1f}% >= {threshold:.1f}% minimum"
                if passed
                else f"Annualized ROI {annualized:.1f}% below {threshold:.1f}% minimum"
            ),
            threshold,
            annualized,
        )


quality_filter = QualityFilterPipeline()
