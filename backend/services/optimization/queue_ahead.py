"""Canonical FIFO queue-position formula — single source of truth for
"how many shares are ahead of my maker order in the queue?"

Used by:
  * :class:`ExecutionEstimator._estimate_maker` — live shadow trader
    fill-probability forecasting + post-run ensemble bracket analytics.
  * :class:`services.backtest.matching_engine.MatchingEngine` — the
    actual backtest fill simulator's resting-order gate.

Keeping one implementation here means backtest realism and live
forecasting can't drift apart: the matcher's queue model is, by
construction, the same one the strategy used to forecast at order
placement time.

The formula encodes the standard Polymarket CLOB price-time priority:

    queue_ahead = better_priced_same_side_total
                + same_price_level_size * maker_queue_ahead_fraction
                * depth_factor

Components:

  * **better_priced_same_side_total** — full size of every same-side
    level at a price that gets filled BEFORE ours.  For a BUY at
    0.50, this is bids at 0.51, 0.52, ...; for a SELL at 0.51, it's
    asks at 0.50, 0.49, ...
  * **maker_queue_ahead_fraction** (default 0.65) — fraction of the
    same-level size we treat as ahead of us.  Reflects that
    time priority within a level is partial information: some of
    the visible depth arrived after us (we go to the back), some
    arrived before (we wait).  0.65 is the empirically-calibrated
    value the live trader uses; backtest must match.
  * **depth_factor** — staleness adjustment.  Books observed more
    than ``max_book_age_ms`` ago decay toward
    ``min_depth_factor`` to model the fact that older snapshots
    over-represent visible depth (some orders cancelled before we
    saw the next update).  At admit time the book is fresh
    (book_age_ms ≈ 0) so depth_factor ≈ 1.0; the backtest matcher
    uses fresh snapshots so this is essentially identity.

If you change the formula here, ``ExecutionEstimator`` and the
backtest matcher both pick it up automatically.  Add a regression
test to :mod:`tests.test_queue_ahead`.
"""
from __future__ import annotations

from typing import Any, Iterable


_MAKER_QUEUE_AHEAD_FRACTION_DEFAULT = 0.65
"""Same-level fraction default; matches ``ExecutionEstimatorConfig``."""


def _coerce_float(value: Any) -> float:
    """Permissive float coercion — mirrors ExecutionEstimator's helper
    so behaviour is identical for malformed level data."""
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _iter_levels(levels: Any) -> Iterable[tuple[float, float]]:
    """Yield (price, size) pairs.  Accepts dicts or attribute objects —
    matches the polymorphic input shape ExecutionEstimator already
    handles (live ws book, backtest PriceLevel dataclass, REST dict)."""
    for lvl in list(levels or []):
        if isinstance(lvl, dict):
            price = _coerce_float(lvl.get("price"))
            size = _coerce_float(lvl.get("size"))
        else:
            price = _coerce_float(getattr(lvl, "price", None))
            size = _coerce_float(getattr(lvl, "size", None))
        if price > 0 and size > 0:
            yield price, size


def compute_queue_ahead_shares(
    *,
    bids: Any,
    asks: Any,
    side: str,
    limit_price: float,
    maker_queue_ahead_fraction: float = _MAKER_QUEUE_AHEAD_FRACTION_DEFAULT,
    depth_factor: float = 1.0,
) -> float:
    """Return the canonical queue-ahead shares for one resting maker.

    ``side`` ∈ {'BUY', 'SELL'} (case-insensitive; ``BUY_YES`` / ``BUY_NO``
    also accepted to match strategy SDK conventions).

    A resting BUY rests on the BID side, so its queue is on bids.  A
    resting SELL rests on the ASK side.  ``better_priced`` for a BUY
    means bids at higher prices than ours (they fill against incoming
    sells first); for a SELL it's asks at lower prices than ours.

    ``depth_factor`` is multiplied in at the end; pass 1.0 when the
    book is fresh (e.g. at admit time in the backtest matcher), or a
    value < 1.0 for stale books (live trader uses
    ``_book_depth_factor(book_age_ms, config)``).
    """
    side_norm = (side or "").upper()
    is_buy = side_norm in ("BUY", "BUY_YES", "BUY_NO")

    # Walk own side; better-priced same-side depth is the cumulative
    # FIFO queue ahead of us, same-level is fractional.
    levels = bids if is_buy else asks
    better_queue = 0.0
    same_level_size = 0.0
    for price, size in _iter_levels(levels):
        if is_buy:
            if price > limit_price + 1e-12:
                better_queue += size
            elif abs(price - limit_price) <= 1e-9:
                same_level_size += size
        else:
            if price < limit_price - 1e-12:
                better_queue += size
            elif abs(price - limit_price) <= 1e-9:
                same_level_size += size

    return (better_queue + same_level_size * maker_queue_ahead_fraction) * depth_factor
