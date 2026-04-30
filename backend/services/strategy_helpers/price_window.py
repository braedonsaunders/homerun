"""Stream-agnostic rolling price window for strategy use.

Replaces the prior :class:`MarketPriceHistory` (which hardcoded yes/no
sides) with a single-stream window that makes no outcome assumptions.
Use one ``PriceWindow`` per price stream you care about — typically
``dict[token_id, PriceWindow]`` for a market's outcomes (so a 3+ outcome
market gets one window per token), or a single ``PriceWindow`` for an
external feed like a crypto spot price or Chainlink oracle.

Exposed via :class:`services.strategy_sdk.StrategySDK` as
``StrategySDK.PriceWindow`` so any strategy can instantiate one without
a direct import.

Typical use::

    from services.strategy_sdk import StrategySDK

    windows: dict[str, StrategySDK.PriceWindow] = {}

    def on_tick(token_id, price, ts_ms):
        w = windows.setdefault(token_id, StrategySDK.PriceWindow(window_seconds=60))
        w.record(price, ts_ms)
        if w.has_data and w.realized_volatility_bps_per_sec() > 50:
            ...

The window is rolling — observations older than ``window_seconds`` are
evicted on every ``record()`` call. Methods that need at least two
observations return None or 0.0 when the window is empty/sparse.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


DEFAULT_WINDOW_SECONDS = 300.0


@dataclass
class PriceWindow:
    """Rolling per-stream price observations.

    ``samples`` holds ``(ts_ms, price)`` tuples in append order. Old
    entries are evicted when ``record()`` runs.
    """

    window_seconds: float = DEFAULT_WINDOW_SECONDS
    samples: deque[tuple[int, float]] = field(default_factory=deque)

    def record(self, price: float, ts_ms: Optional[int] = None) -> None:
        """Append a price observation and evict any older than the window.

        ``ts_ms`` defaults to wall-clock now (``time.time() * 1000``) so
        callers with a server-supplied timestamp can override.
        """
        ts = int(ts_ms) if ts_ms is not None else int(time.time() * 1000)
        self.samples.append((ts, float(price)))
        self._evict(ts)

    def _evict(self, now_ms: int) -> None:
        cutoff = now_ms - int(self.window_seconds * 1000)
        while self.samples and self.samples[0][0] < cutoff:
            self.samples.popleft()

    def latest(self) -> Optional[tuple[int, float]]:
        """Return the most recent ``(ts_ms, price)``, or None when empty."""
        if not self.samples:
            return None
        return self.samples[-1]

    def at_or_before(self, ts_ms: int) -> Optional[tuple[int, float]]:
        """Return the most recent observation with ``timestamp <= ts_ms``.

        Returns None when no observation in the window is at-or-before
        the requested time. Walks samples in reverse order so amortized
        cost is O(1) for recent timestamps.
        """
        for sample_ts, sample_price in reversed(self.samples):
            if sample_ts <= ts_ms:
                return (sample_ts, sample_price)
        return None

    def log_return(self, seconds_ago: float) -> Optional[float]:
        """Return ``log(latest / prior)`` over a ``seconds_ago`` window.

        Picks the prior observation via :meth:`at_or_before`. Returns
        None when the window is empty, ``seconds_ago`` is non-positive,
        or there's no observation old enough to anchor the return.
        """
        if not self.samples or seconds_ago <= 0:
            return None
        latest_ts, latest_price = self.samples[-1]
        prior = self.at_or_before(latest_ts - int(seconds_ago * 1000))
        if prior is None:
            return None
        prior_price = prior[1]
        if prior_price <= 0 or latest_price <= 0:
            return None
        return math.log(latest_price / prior_price)

    def stddev(self) -> float:
        """Return the standard deviation of recent prices.

        Returns 0.0 when the window has fewer than 2 samples.
        """
        if len(self.samples) < 2:
            return 0.0
        prices = [p for _, p in self.samples]
        mean = sum(prices) / len(prices)
        var = sum((p - mean) ** 2 for p in prices) / len(prices)
        return math.sqrt(var)

    def realized_volatility_bps_per_sec(self, sample_period_s: float = 1.0) -> float:
        """Realized volatility, normalized to bps per ``sample_period_s``.

        For each adjacent pair of samples, computes the log return scaled
        to a standard ``sample_period_s`` window (so heterogeneous tick
        spacing produces comparable returns), then returns the std-dev
        of those scaled returns in basis points (1 bp = 0.0001).

        Returns 0.0 when the window has fewer than 2 samples or
        ``sample_period_s`` is non-positive.
        """
        if len(self.samples) < 2 or sample_period_s <= 0:
            return 0.0
        scaled_returns: list[float] = []
        prev_ts, prev_p = self.samples[0]
        for ts, p in list(self.samples)[1:]:
            if prev_p > 0 and p > 0:
                dt_s = max(0.001, (ts - prev_ts) / 1000.0)
                r = math.log(p / prev_p)
                scaled = r * math.sqrt(sample_period_s / dt_s)
                scaled_returns.append(scaled)
            prev_ts, prev_p = ts, p
        if not scaled_returns:
            return 0.0
        mean = sum(scaled_returns) / len(scaled_returns)
        var = sum((r - mean) ** 2 for r in scaled_returns) / len(scaled_returns)
        return math.sqrt(var) * 10_000.0

    def distance_bps(self, reference_price: float) -> Optional[float]:
        """Distance from the latest price to ``reference_price`` in bps.

        Positive when latest is above the reference, negative below.
        Returns None when the window is empty or ``reference_price`` is
        non-positive.
        """
        if not self.samples or reference_price <= 0:
            return None
        latest_price = self.samples[-1][1]
        return ((latest_price - reference_price) / reference_price) * 10_000.0

    @property
    def has_data(self) -> bool:
        """True when the window has at least 2 samples (statistics meaningful)."""
        return len(self.samples) >= 2

    def __len__(self) -> int:
        return len(self.samples)


# ---------------------------------------------------------------------------
# MultiWindow — fan one tick into N rolling lookbacks on a shared price stream
# ---------------------------------------------------------------------------


class MultiWindow:
    """Fan a single price stream into multiple rolling lookback windows.

    Removes the dict-of-dicts boilerplate when a strategy wants to reason
    about the same price across several timeframes (e.g. 5m / 15m / 1h /
    4h). One ``record(price)`` updates every child window; per-window
    statistics are then queryable by label.

    Typical use::

        from services.strategy_sdk import StrategySDK

        mw = StrategySDK.MultiWindow(lookbacks={
            "5m":  300,
            "15m": 900,
            "1h":  3600,
            "4h":  14400,
        })

        def on_tick(price):
            mw.record(price)
            if mw.all_agree(direction="up", min_return=0.01):
                ...   # all four lookbacks confirm an upward move >= 1%

    Notes
    -----
    * ``lookbacks`` maps a label -> seconds. Window sizes are independent
      and the sample buffers are separate per label, so the longest
      window does not bound the shortest one's eviction policy.
    * ``log_returns()`` returns one log-return per label, computed against
      a sample ``seconds_ago=window_seconds`` ago — i.e. the full
      lookback. ``None`` entries indicate insufficient data for that
      label (rather than dropped entries) so callers can distinguish
      "haven't seen enough ticks" from "no movement."
    """

    __slots__ = ("_windows",)

    def __init__(self, lookbacks: dict[str, float]) -> None:
        if not lookbacks:
            raise ValueError("MultiWindow requires at least one lookback")
        windows: dict[str, PriceWindow] = {}
        for label, seconds in lookbacks.items():
            secs = float(seconds)
            if secs <= 0:
                raise ValueError(f"MultiWindow lookback '{label}' must be > 0 (got {seconds})")
            windows[str(label)] = PriceWindow(window_seconds=secs)
        self._windows = windows

    # ── Recording ──────────────────────────────────────────────

    def record(self, price: float, ts_ms: Optional[int] = None) -> None:
        """Append the same observation to every child window."""
        ts = int(ts_ms) if ts_ms is not None else int(time.time() * 1000)
        for window in self._windows.values():
            window.record(price, ts_ms=ts)

    # ── Access ─────────────────────────────────────────────────

    def __getitem__(self, label: str) -> PriceWindow:
        return self._windows[str(label)]

    def __contains__(self, label: object) -> bool:
        return str(label) in self._windows

    def __iter__(self):
        return iter(self._windows)

    def __len__(self) -> int:
        return len(self._windows)

    def labels(self) -> list[str]:
        """Lookback labels in insertion order."""
        return list(self._windows.keys())

    def windows(self) -> dict[str, PriceWindow]:
        """Live mapping of label -> PriceWindow (mutating returned dict
        does not detach windows from the MultiWindow)."""
        return dict(self._windows)

    @property
    def has_data(self) -> bool:
        """True when **every** child window has enough samples for statistics."""
        return all(w.has_data for w in self._windows.values())

    # ── Aggregate stats ────────────────────────────────────────

    def log_returns(self) -> dict[str, Optional[float]]:
        """Per-label log return computed over each window's full lookback.

        Returns ``None`` for labels that lack a prior observation old
        enough to anchor the return.
        """
        return {
            label: window.log_return(seconds_ago=window.window_seconds)
            for label, window in self._windows.items()
        }

    def realized_volatility_bps_per_sec(
        self, sample_period_s: float = 1.0
    ) -> dict[str, float]:
        """Per-label realized volatility in bps per ``sample_period_s``."""
        return {
            label: window.realized_volatility_bps_per_sec(sample_period_s)
            for label, window in self._windows.items()
        }

    # ── Multi-window confirmation primitives ───────────────────

    def aligned_count(
        self,
        *,
        direction: str = "up",
        min_return: float = 0.0,
    ) -> int:
        """Number of lookbacks whose log-return matches ``direction``.

        ``direction`` is ``"up"`` or ``"down"``. ``min_return`` is the
        minimum |log-return| (e.g. ``0.01`` for ~1%) to count toward
        agreement; defaults to 0 (any sign-aligned movement counts).
        Labels without enough data contribute nothing.
        """
        sign = _direction_sign(direction)
        threshold = abs(float(min_return))
        count = 0
        for ret in self.log_returns().values():
            if ret is None:
                continue
            if sign > 0 and ret >= threshold:
                count += 1
            elif sign < 0 and ret <= -threshold:
                count += 1
        return count

    def all_agree(
        self,
        *,
        direction: str = "up",
        min_return: float = 0.0,
    ) -> bool:
        """True iff every lookback that has data agrees on ``direction``.

        Requires every child window to ``has_data``; this avoids reporting
        agreement on a partially-populated tracker.
        """
        if not self.has_data:
            return False
        return self.aligned_count(direction=direction, min_return=min_return) == len(
            self._windows
        )


# ---------------------------------------------------------------------------
# Module-level confirmation helpers (work on plain dicts so callers can mix
# values from different sources, not just MultiWindow)
# ---------------------------------------------------------------------------


def _direction_sign(direction: str) -> int:
    norm = str(direction or "").strip().lower()
    if norm in {"up", "long", "buy", "yes", "+", "1"}:
        return 1
    if norm in {"down", "short", "sell", "no", "-", "-1"}:
        return -1
    raise ValueError(f"Unknown direction '{direction}' — expected 'up' or 'down'")


def timeframes_agree(
    returns_by_label: dict[str, Optional[float]],
    *,
    direction: str = "up",
    min_count: int = 1,
    min_return: float = 0.0,
) -> bool:
    """Return True when at least ``min_count`` labels agree on ``direction``.

    ``returns_by_label`` is the kind of dict returned by
    :meth:`MultiWindow.log_returns` — values may be ``None`` to indicate
    insufficient data. ``min_return`` is the minimum |log-return| each
    label must reach to count toward agreement (defaults to 0, i.e. any
    sign-aligned movement counts).
    """
    sign = _direction_sign(direction)
    threshold = abs(float(min_return))
    if min_count < 1:
        raise ValueError("min_count must be >= 1")
    count = 0
    for ret in returns_by_label.values():
        if ret is None:
            continue
        if sign > 0 and ret >= threshold:
            count += 1
        elif sign < 0 and ret <= -threshold:
            count += 1
        if count >= min_count:
            return True
    return False


def weighted_signal(
    returns_by_label: dict[str, Optional[float]],
    weights: dict[str, float],
) -> Optional[float]:
    """Weighted average of per-label log-returns.

    Labels missing from ``returns_by_label`` (or with ``None`` value)
    are skipped, and weights are renormalised over the labels that
    contributed — so a partially-populated MultiWindow still produces a
    meaningful signal rather than collapsing to zero.

    Returns ``None`` when no label contributes (all missing or all
    weights are zero).
    """
    total_weight = 0.0
    weighted_sum = 0.0
    for label, weight in weights.items():
        w = float(weight)
        if w <= 0:
            continue
        ret = returns_by_label.get(label)
        if ret is None:
            continue
        weighted_sum += w * float(ret)
        total_weight += w
    if total_weight <= 0:
        return None
    return weighted_sum / total_weight
