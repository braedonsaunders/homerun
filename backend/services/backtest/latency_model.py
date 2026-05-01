"""Parameterized latency model for the backtest matching engine.

In live trading two latencies matter:

1. **Submit latency** — wall-clock between the strategy emitting an order
   and the venue acknowledging it. During this window the book has moved.
2. **Cancel latency** — wall-clock between issuing a cancel and the venue
   removing the order from the book. During this window the order may
   still match incoming orders.

Polymarket-via-CLOB Gateway typically sees ~200–600ms submit latency end
to end (Python serialization + EIP-712 signing + provider RPC + CLOB API).
We model both as independent log-normal samples with a configurable scale,
so the same backtest can be run at p50, p95, or p99 latency profiles.

This is **calibratable**: when we begin recording submit→ack and
ack→cancel deltas in production, ``LatencyProfile.from_observations`` will
estimate (mu, sigma) from a sample of measurements.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Iterable, Optional


@dataclass(frozen=True)
class LatencyProfile:
    """Parameters of a log-normal latency distribution (in milliseconds).

    Use the class methods to construct from intuitive p50/p95 values rather
    than working in (mu, sigma) directly.
    """

    mu_ms: float
    sigma_ms: float
    minimum_ms: float = 1.0
    maximum_ms: float = 10_000.0

    @classmethod
    def from_quantiles(
        cls,
        *,
        p50_ms: float,
        p95_ms: float,
        minimum_ms: float = 1.0,
        maximum_ms: float = 10_000.0,
    ) -> "LatencyProfile":
        """Solve (mu, sigma) for a log-normal that hits a given p50/p95.

        log-normal: ln(X) ~ N(mu, sigma^2)
        median = exp(mu)            => mu = ln(p50)
        p95    = exp(mu + 1.645*sigma) => sigma = (ln(p95) - mu) / 1.645
        """
        if p50_ms <= 0 or p95_ms <= 0 or p95_ms <= p50_ms:
            raise ValueError(
                f"latency quantiles invalid: p50={p50_ms} p95={p95_ms}"
            )
        mu = math.log(p50_ms)
        sigma = (math.log(p95_ms) - mu) / 1.6448536269514722
        return cls(mu_ms=mu, sigma_ms=max(1e-6, sigma), minimum_ms=minimum_ms, maximum_ms=maximum_ms)

    @classmethod
    def from_observations(
        cls,
        samples_ms: Iterable[float],
        *,
        minimum_ms: float = 1.0,
        maximum_ms: float = 10_000.0,
    ) -> "LatencyProfile":
        """Fit log-normal MLE from observed millisecond samples."""
        positive = [float(s) for s in samples_ms if s > 0]
        if len(positive) < 5:
            raise ValueError(
                f"need >= 5 positive samples to fit, got {len(positive)}"
            )
        logs = [math.log(s) for s in positive]
        mu = sum(logs) / len(logs)
        var = sum((x - mu) ** 2 for x in logs) / max(1, len(logs) - 1)
        sigma = math.sqrt(max(var, 1e-9))
        return cls(mu_ms=mu, sigma_ms=sigma, minimum_ms=minimum_ms, maximum_ms=maximum_ms)

    def quantile(self, q: float) -> float:
        """Inverse CDF for the log-normal at quantile ``q`` in (0, 1)."""
        if not (0.0 < q < 1.0):
            raise ValueError(f"q must be in (0, 1), got {q}")
        z = _inverse_normal_cdf(q)
        ms = math.exp(self.mu_ms + self.sigma_ms * z)
        return _clamp(ms, self.minimum_ms, self.maximum_ms)

    def sample(self, rng: random.Random) -> float:
        """Draw one millisecond sample from the distribution."""
        z = rng.gauss(0.0, 1.0)
        ms = math.exp(self.mu_ms + self.sigma_ms * z)
        return _clamp(ms, self.minimum_ms, self.maximum_ms)


# ── Sensible defaults calibrated against typical Polymarket CLOB ───────
# These approximate p50=350ms, p95=900ms for submit; cancel is typically
# faster (~200ms / 600ms) because no signing round trip is needed.

DEFAULT_SUBMIT_PROFILE = LatencyProfile.from_quantiles(p50_ms=350.0, p95_ms=900.0)
DEFAULT_CANCEL_PROFILE = LatencyProfile.from_quantiles(p50_ms=200.0, p95_ms=600.0)


@dataclass
class LatencyModel:
    """Pair of latency distributions for submit and cancel, with
    optional multi-leg correlation.

    The matching engine consumes ``LatencyModel.sample_submit_ms`` /
    ``sample_cancel_ms`` to advance simulated time after each interaction.
    Pass a fixed seed for reproducible backtests.

    Multi-leg correlation: when ``correlation_window_ms > 0``, orders
    submitted within the same window share a single latency draw.
    The institutional reality this models: when a strategy fires 4
    legs of an arbitrage in the same tick, all 4 traverse the same
    network path and the same RPC queue — their latencies are heavily
    correlated, not independent.  The default of 5ms reflects the
    typical resolution of "same scheduling tick" on Polymarket's
    CLOB.  Set to 0 to restore the legacy independent-draw behavior.

    Pass an emit timestamp via ``sample_submit_ms(at=ts)`` to bucket
    correlated calls; calling without ``at=`` always draws independently
    (legacy behavior, used by single-leg strategies and tests).
    """

    submit: LatencyProfile = field(default_factory=lambda: DEFAULT_SUBMIT_PROFILE)
    cancel: LatencyProfile = field(default_factory=lambda: DEFAULT_CANCEL_PROFILE)
    rng: Optional[random.Random] = None
    seed: Optional[int] = None
    correlation_window_ms: float = 5.0

    def __post_init__(self) -> None:
        if self.rng is None:
            self.rng = random.Random(self.seed)
        # Correlation cache: {("submit"|"cancel", bucket_id): cached_ms}.
        # Bucket id is the floor(ts_ms / window_ms) so all calls in the
        # same window resolve to the same dict entry.  Bounded to the
        # most recent 1024 buckets to avoid unbounded growth.
        self._bucket_cache: dict[tuple[str, int], float] = {}
        self._bucket_order: list[tuple[str, int]] = []

    def _bucket_id(self, at) -> Optional[int]:
        if at is None or self.correlation_window_ms <= 0:
            return None
        try:
            ts_ms = float(at.timestamp()) * 1000.0
        except AttributeError:
            try:
                ts_ms = float(at) * 1000.0
            except (TypeError, ValueError):
                return None
        return int(ts_ms // float(self.correlation_window_ms))

    def _correlated_sample(self, kind: str, profile: "LatencyProfile", at) -> float:
        bucket = self._bucket_id(at)
        if bucket is None:
            return profile.sample(self.rng)  # type: ignore[arg-type]
        key = (kind, bucket)
        cached = self._bucket_cache.get(key)
        if cached is not None:
            return cached
        drawn = profile.sample(self.rng)  # type: ignore[arg-type]
        self._bucket_cache[key] = drawn
        self._bucket_order.append(key)
        if len(self._bucket_order) > 1024:
            old = self._bucket_order.pop(0)
            self._bucket_cache.pop(old, None)
        return drawn

    def sample_submit_ms(self, *, at=None) -> float:
        if at is None:
            return self.submit.sample(self.rng)  # type: ignore[arg-type]
        return self._correlated_sample("submit", self.submit, at)

    def sample_cancel_ms(self, *, at=None) -> float:
        if at is None:
            return self.cancel.sample(self.rng)  # type: ignore[arg-type]
        return self._correlated_sample("cancel", self.cancel, at)

    @classmethod
    def deterministic(
        cls, *, submit_ms: float = 350.0, cancel_ms: float = 200.0
    ) -> "LatencyModel":
        """Constant-latency model for unit tests."""
        # sigma=0 collapses log-normal to a point mass at exp(mu)
        return cls(
            submit=LatencyProfile(mu_ms=math.log(submit_ms), sigma_ms=1e-9),
            cancel=LatencyProfile(mu_ms=math.log(cancel_ms), sigma_ms=1e-9),
        )


# ── Helpers ──────────────────────────────────────────────────────────────


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def _inverse_normal_cdf(p: float) -> float:
    """Beasley–Springer/Moro approximation to the standard normal inverse CDF.

    Accurate to ~1e-9 across the whole (0, 1) interval. We avoid scipy as
    a dependency.
    """
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]
    p_low = 0.02425
    p_high = 1 - p_low
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        )
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
        )
    q = math.sqrt(-2 * math.log(1 - p))
    return -(
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    )


__all__ = [
    "LatencyProfile",
    "LatencyModel",
    "DEFAULT_SUBMIT_PROFILE",
    "DEFAULT_CANCEL_PROFILE",
]
