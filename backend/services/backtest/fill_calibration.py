"""Fill-model calibration — execution accuracy as an error band.

The honest split this enforces:

  * **Signal parity** (``decision_parity``) — should approach 100% once
    the input-recording gaps are closed, because a strategy IS a pure
    function of its recorded inputs.
  * **Execution accuracy** (this module) — can NEVER be 100%.  Your own
    order perturbs the book, so fills cannot be replayed from a static
    recorded book; the backtester MODELS them (matching engine / fill
    simulator / venue model / Cox).  This module measures how well that
    model matches reality by comparing its assumptions against the
    ``order.fill`` topic the live fill-monitor records, and reports an
    error band — not a pass/fail.

Until the fill tee has accumulated live fills over a window, the realized
side is empty and the report says so rather than implying a calibrated
model.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _pct(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = max(0, min(len(sorted_vals) - 1, int(p * (len(sorted_vals) - 1))))
    return float(sorted_vals[idx])


@dataclass
class FillCalibrationReport:
    window_start: datetime
    window_end: datetime
    realized_n: int = 0
    realized_fill_rate_mean: float = 0.0   # mean filled fraction (0..1)
    realized_partial_rate: float = 0.0     # fraction of fills that were partial
    realized_fee_mean: float = 0.0
    fill_percent_p10: float = 0.0
    fill_percent_p50: float = 0.0
    fill_percent_p90: float = 0.0
    model_fill_rate: Optional[float] = None
    fill_rate_abs_error: Optional[float] = None
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        base = (
            f"[fill-calibration {self.window_start:%Y-%m-%d %H:%M}"
            f"..{self.window_end:%H:%M}] realized_n={self.realized_n} "
            f"fill_rate={self.realized_fill_rate_mean:.3f} "
            f"partial_rate={self.realized_partial_rate:.3f} "
            f"fee={self.realized_fee_mean:.4f}"
        )
        if self.model_fill_rate is not None:
            base += (
                f" | model_fill_rate={self.model_fill_rate:.3f} "
                f"abs_error={self.fill_rate_abs_error:.3f}"
            )
        return base


async def recorded_fills(
    *, start: datetime, end: datetime, token_ids: Optional[list[str]] = None
) -> list[dict[str, Any]]:
    """Load realized live fills from the ``order.fill`` topic."""
    import services.recorded_event_bus.storage  # noqa: F401  attach storage
    from services.recorded_event_bus.bus import bus, ReplayWindow
    from services.recorded_event_bus.decision_recorder import ORDER_FILL_TOPIC

    def _us(dt: datetime) -> int:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1_000_000)

    ef = None
    if token_ids:
        ef = {ORDER_FILL_TOPIC: frozenset(str(t) for t in token_ids)}
    win = ReplayWindow(
        start_us=_us(start), end_us=_us(end),
        topics=(ORDER_FILL_TOPIC,), entity_filter=ef,
    )
    out: list[dict[str, Any]] = []
    async for ev in bus.replay(win):
        out.append(dict(ev.payload))
    return out


async def calibration_report(
    *,
    start: datetime,
    end: datetime,
    token_ids: Optional[list[str]] = None,
    model_predicted_fill_rate: Optional[float] = None,
) -> FillCalibrationReport:
    """Compute realized fill stats and, when a model prediction is given,
    the absolute error band between the backtest fill model and reality."""
    report = FillCalibrationReport(window_start=start, window_end=end)
    fills = await recorded_fills(start=start, end=end, token_ids=token_ids)
    report.realized_n = len(fills)

    if not fills:
        report.notes.append(
            "no recorded live fills in window — the fill tee hasn't run live "
            "over this window yet, so the venue/fill model is UNCALIBRATED. "
            "Execution-accuracy claims for this window are model-only."
        )
        if model_predicted_fill_rate is not None:
            report.model_fill_rate = float(model_predicted_fill_rate)
        return report

    def _f(v: Any, d: float = 0.0) -> float:
        try:
            return float(v)
        except (TypeError, ValueError):
            return d

    fill_fracs = [max(0.0, min(1.0, _f(f.get("fill_percent")) / 100.0)) for f in fills]
    partials = [1.0 if _f(f.get("fill_percent")) < 99.9 else 0.0 for f in fills]
    fees = [_f(f.get("fee")) for f in fills]
    pcts = sorted(_f(f.get("fill_percent")) for f in fills)

    report.realized_fill_rate_mean = sum(fill_fracs) / len(fill_fracs)
    report.realized_partial_rate = sum(partials) / len(partials)
    report.realized_fee_mean = sum(fees) / len(fees)
    report.fill_percent_p10 = _pct(pcts, 0.1)
    report.fill_percent_p50 = _pct(pcts, 0.5)
    report.fill_percent_p90 = _pct(pcts, 0.9)

    if model_predicted_fill_rate is not None:
        report.model_fill_rate = float(model_predicted_fill_rate)
        report.fill_rate_abs_error = abs(
            report.model_fill_rate - report.realized_fill_rate_mean
        )
        report.notes.append(
            "execution accuracy is an ERROR BAND, not parity: realized fills "
            "are venue-dependent and your own order perturbs the book. Treat "
            "backtest PnL as carrying this fill-rate error, separate from "
            "signal parity (see services.backtest.decision_parity)."
        )
    return report
