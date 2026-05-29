"""Learned empirical constants for the fill estimator.

Replaces the hardcoded magic numbers in ExecutionEstimatorConfig
(``displayed_depth_factor=0.88``, ``maker_queue_ahead_fraction=0.65``,
etc.) with values estimated from the actual canonical book-delta feed
(``DELTA_SCHEMA`` parquet, via ``marketdata.aggregate_delta_events``).

Each constant has:

* a measurement function that crunches canonical delta parquet / fill events
* a 5-minute in-process cache so the order hot path doesn't re-scan parquet
* a fallback to the current default if no data is available yet

These constants are *additionally* exposed in the UI under
``Strategies → ML Models → Fill Model``, so an operator can override
any of them.  The override takes priority over the measured value;
setting the override to None reverts to "use measured".
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import asyncio

from sqlalchemy.ext.asyncio import AsyncSession



logger = logging.getLogger("fill_simulator.empirical_constants")


# How often to refresh the empirical constants from the DB.  Trade /
# cancel ratios change slowly (hours); 15 min is a sane refresh.
_REFRESH_INTERVAL_SECONDS = 900.0


@dataclass
class EmpiricalConstants:
    # Default values are the same as the current hardcoded ones, but
    # tagged ``measured=False`` until the first DB refresh succeeds.
    displayed_depth_factor: float = 0.88
    maker_queue_ahead_fraction: float = 0.65
    maker_trade_flow_multiplier: float = 1.20
    adverse_selection_multiplier: float = 0.70
    stale_depth_decay: float = 0.55
    min_depth_factor: float = 0.20

    measured: bool = False
    sample_count: int = 0
    measured_at_epoch: float = 0.0
    notes: str = "default — no measured data yet"


@dataclass
class _CacheState:
    constants: EmpiricalConstants = field(default_factory=EmpiricalConstants)
    last_refresh_epoch: float = 0.0
    overrides: dict[str, float | None] = field(default_factory=dict)


_state = _CacheState()


def get_empirical_constants() -> EmpiricalConstants:
    """Sync read.  Returns the current cached constants — overrides
    applied — without hitting the DB.  Call ``refresh_async()`` from
    a background loop to keep them fresh."""
    base = _state.constants
    if not _state.overrides:
        return base
    out = EmpiricalConstants(
        displayed_depth_factor=_state.overrides.get("displayed_depth_factor") or base.displayed_depth_factor,
        maker_queue_ahead_fraction=_state.overrides.get("maker_queue_ahead_fraction") or base.maker_queue_ahead_fraction,
        maker_trade_flow_multiplier=_state.overrides.get("maker_trade_flow_multiplier") or base.maker_trade_flow_multiplier,
        adverse_selection_multiplier=_state.overrides.get("adverse_selection_multiplier") or base.adverse_selection_multiplier,
        stale_depth_decay=_state.overrides.get("stale_depth_decay") or base.stale_depth_decay,
        min_depth_factor=_state.overrides.get("min_depth_factor") or base.min_depth_factor,
        measured=base.measured,
        sample_count=base.sample_count,
        measured_at_epoch=base.measured_at_epoch,
        notes=base.notes,
    )
    return out


def set_override(key: str, value: float | None) -> None:
    """UI hook.  ``value=None`` reverts to measured."""
    if key not in {
        "displayed_depth_factor",
        "maker_queue_ahead_fraction",
        "maker_trade_flow_multiplier",
        "adverse_selection_multiplier",
        "stale_depth_decay",
        "min_depth_factor",
    }:
        raise ValueError(f"unknown empirical constant override: {key}")
    if value is None:
        _state.overrides.pop(key, None)
        return
    if not (0.0 < float(value) <= 5.0):
        raise ValueError(f"{key} must be in (0, 5]")
    _state.overrides[key] = float(value)


def get_overrides() -> dict[str, float]:
    return dict(_state.overrides)


async def refresh_async(*, session: AsyncSession | None = None) -> EmpiricalConstants:
    """Recompute the constants from the canonical book-delta parquet.

    The math is a pragmatic first cut — empirically validate further
    once we have real data flowing through it:

    * **displayed_depth_factor**: across all trade events, what
      fraction of the queue_depth_before was actually traded vs
      remained as displayed-but-not-traded?  Empirical visible-
      liquidity ratio.
    * **stale_depth_decay**: the same ratio computed on cancel
      events — how much of a level "decays" purely from cancellation
      pressure when stale.
    * **maker_queue_ahead_fraction**: assumed prior on where in the
      same-level queue you sit.  Computed as the median fraction of
      same-level depth that cleared before each level was depleted.
      No clean closed-form yet — we keep the default of 0.65 for V1
      and rely on the Cox covariate ``queue_ahead_shares`` (Phase 2)
      to account for variance.
    """
    # ``session`` is accepted for backwards compatibility but no longer used:
    # book deltas live in the canonical parquet plane, not SQL.  Aggregate the
    # last 24h of trade/cancel events from parquet off the event loop.
    try:
        from services.marketdata.deltas import aggregate_delta_events

        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=24)
        agg = await asyncio.wait_for(
            asyncio.to_thread(aggregate_delta_events, start=cutoff, end=now),
            timeout=20.0,
        )
    except (asyncio.TimeoutError, Exception) as exc:
        logger.warning(
            "Empirical-constants aggregate timed out / failed; keeping cached values: %s",
            exc,
        )
        _state.last_refresh_epoch = time.monotonic()
        return _state.constants
    try:
        n_trade = int(agg.get("n_trade", 0) or 0)
        n_cancel = int(agg.get("n_cancel", 0) or 0)
        total = n_trade + n_cancel
        if total < 50:
            # Not enough decomposed events yet; keep defaults.
            _state.last_refresh_epoch = time.monotonic()
            return _state.constants

        # Of all observed depth disappearance, fraction that was a
        # genuine fill (not cancel).  This is the empirical
        # "visibility" of the displayed book — high trade fraction =
        # tight, real liquidity; high cancel fraction = spoofy book.
        trade_fraction = n_trade / total

        # Sum-of-sizes weighted versions came from the same scan.
        trade_sz = float(agg.get("trade_size_sum", 0) or 0)
        cancel_sz = float(agg.get("cancel_size_sum", 0) or 0)
        total_sz = trade_sz + cancel_sz
        size_trade_fraction = trade_sz / total_sz if total_sz > 0 else trade_fraction

        # The displayed_depth_factor is "what fraction of stated depth
        # is actually fillable" — proxy as the size-trade fraction
        # bounded to [0.4, 0.99].  A perfectly real book has factor
        # close to 1; a heavily-spoofed one has factor near 0.4.
        ddf = max(0.40, min(0.99, size_trade_fraction))

        # The stale_depth_decay is how much of a level decays per
        # second of staleness purely from cancel pressure.  Rough
        # estimate: cancel_size / total over a 1s window normalized
        # against the typical book depth — but for V1 we use the
        # cancel fraction directly, scaled to a 0..0.99 band so the
        # estimator math stays well-behaved.
        cancel_fraction = 1.0 - size_trade_fraction
        sdd = max(0.05, min(0.99, cancel_fraction))

        # min_depth_factor (the floor when a book is maximally stale)
        # is roughly half the ddf — empirical floor in our recorded
        # 5 days of microstructure data.  Will be refined later once
        # we cross-reference with book staleness explicitly.
        mdf = max(0.05, min(0.5, ddf * 0.5))

        # The other two stay as priors for V1 — Cox covariates handle
        # the variance.
        constants = EmpiricalConstants(
            displayed_depth_factor=ddf,
            maker_queue_ahead_fraction=0.65,
            maker_trade_flow_multiplier=1.20,
            adverse_selection_multiplier=max(0.40, min(0.95, 1.0 - cancel_fraction * 0.5)),
            stale_depth_decay=sdd,
            min_depth_factor=mdf,
            measured=True,
            sample_count=total,
            measured_at_epoch=time.time(),
            notes=(
                f"measured from {total} book delta events / {total_sz:.0f} shares "
                f"({trade_fraction*100:.1f}% trades by count, {size_trade_fraction*100:.1f}% by size)"
            ),
        )
        _state.constants = constants
        _state.last_refresh_epoch = time.monotonic()
        return constants
    except Exception:
        logger.exception("Empirical constants computation failed; keeping cached values")
        _state.last_refresh_epoch = time.monotonic()
        return _state.constants


def time_since_refresh_seconds() -> float:
    if _state.last_refresh_epoch <= 0:
        return float("inf")
    return time.monotonic() - _state.last_refresh_epoch


async def refresh_if_stale() -> EmpiricalConstants:
    """Call from a periodic loop (or hot path before estimator)."""
    if time_since_refresh_seconds() > _REFRESH_INTERVAL_SECONDS:
        try:
            await refresh_async()
        except Exception:
            logger.exception("Empirical constants refresh failed")
    return get_empirical_constants()
