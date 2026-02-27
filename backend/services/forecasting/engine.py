"""Pure-math Bayesian evidence aggregation engine.

Ported from Polyseer's forecasting module (evidence.ts, aggregator.ts,
math.ts, cluster.ts).  Zero external deps — no DB, no HTTP, no AI.
Fully stateless and unit-testable.
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timezone

from .types import ClusterMeta, Evidence, ForecastResult, InfluenceItem

# ---------------------------------------------------------------------------
# Constants (faithful port from Polyseer evidence.ts + aggregator.ts)
# ---------------------------------------------------------------------------

TYPE_CAPS: dict[str, float] = {"A": 1.0, "B": 0.6, "C": 0.3, "D": 0.2}
WEIGHTS = {"v": 0.45, "r": 0.25, "u": 0.15, "t": 0.15}
FIRST_REPORT_PENALTY = 1.0
RECENCY_HALF_LIFE_DAYS = 120.0
DEFAULT_CLUSTER_RHO = 0.6
TRIM_FRACTION = 0.2
MARKET_BLEND_ALPHA = 0.1

# ---------------------------------------------------------------------------
# Math primitives
# ---------------------------------------------------------------------------


def clamp(x: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, x))


def logit(p: float) -> float:
    p = clamp(p, 1e-9, 1.0 - 1e-9)
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def trimmed_mean(values: list[float], fraction: float = 0.2) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    t = max(0, int(fraction * len(sorted_v)))
    trimmed = sorted_v[t : len(sorted_v) - t]
    if not trimmed:
        trimmed = sorted_v
    return sum(trimmed) / len(trimmed)


# ---------------------------------------------------------------------------
# Evidence scoring
# ---------------------------------------------------------------------------


def recency_score(published_at: datetime | None, now: datetime | None = None) -> float:
    if published_at is None:
        return 0.5  # unknown recency -> neutral
    if now is None:
        now = datetime.now(timezone.utc)
    if published_at.tzinfo is None:
        published_at = published_at.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    seconds = max(0.0, (now - published_at).total_seconds())
    days = seconds / 86400.0
    score = 1.0 / (1.0 + days / RECENCY_HALF_LIFE_DAYS)
    return clamp(score, 0.0, 1.0)


def r_from_corroborations(k: int, k0: float = 1.0) -> float:
    return 1.0 - math.exp(-k0 * max(0, k))


def evidence_log_lr(e: Evidence, now: datetime | None = None) -> float:
    if e.log_lr_hint is not None:
        return e.log_lr_hint
    cap = TYPE_CAPS.get(e.type, 0.2)
    ver = clamp(e.verifiability, 0.0, 1.0)
    cons = clamp(e.consistency, 0.0, 1.0)
    r = r_from_corroborations(e.corroborations_indep)
    t = recency_score(e.published_at, now)
    val = e.polarity * cap * (
        WEIGHTS["v"] * ver
        + WEIGHTS["r"] * r
        + WEIGHTS["u"] * cons
        + WEIGHTS["t"] * t
    )
    if e.first_report:
        val *= FIRST_REPORT_PENALTY
    return clamp(val, -cap, cap)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


def cluster_by_origin(evidence: list[Evidence]) -> dict[str, list[Evidence]]:
    clusters: dict[str, list[Evidence]] = defaultdict(list)
    for ev in evidence:
        clusters[ev.origin_id].append(ev)
    return dict(clusters)


def effective_count(m: int, rho: float) -> float:
    r = clamp(rho, 0.0, 0.99)
    return m / (1.0 + (m - 1) * r)


# ---------------------------------------------------------------------------
# Core aggregation
# ---------------------------------------------------------------------------


def aggregate_neutral(
    p0: float,
    evidence: list[Evidence],
    rho_by_cluster: dict[str, float] | None = None,
    trim_fraction: float = TRIM_FRACTION,
) -> tuple[float, list[ClusterMeta], list[InfluenceItem]]:
    """Aggregate evidence into a neutral posterior probability.

    Returns (p_neutral, cluster_metadata, influence_items).
    """
    clusters = cluster_by_origin(evidence)
    log_odds = logit(p0)
    meta: list[ClusterMeta] = []
    contrib: dict[str, float] = {}

    for cid, items in clusters.items():
        m = len(items)
        rho = (rho_by_cluster or {}).get(cid, DEFAULT_CLUSTER_RHO if m > 1 else 0.0)
        m_eff = effective_count(m, rho)
        llrs = [evidence_log_lr(e) for e in items]
        mean_llr = trimmed_mean(llrs, trim_fraction)
        c = m_eff * mean_llr
        contrib[cid] = c
        log_odds += c
        meta.append(ClusterMeta(
            cluster_id=cid,
            size=m,
            rho=rho,
            effective_count=m_eff,
            mean_log_lr=mean_llr,
            contribution=c,
        ))

    p_neutral = sigmoid(log_odds)

    # Leave-one-out influence per evidence item
    influence: list[InfluenceItem] = []
    for cid, items in clusters.items():
        for ev in items:
            others = [x for x in items if x.id != ev.id]
            alt = 0.0
            if others:
                rho_alt = (rho_by_cluster or {}).get(
                    cid, DEFAULT_CLUSTER_RHO if len(others) > 1 else 0.0
                )
                m_eff_alt = effective_count(len(others), rho_alt)
                mean_alt = trimmed_mean(
                    [evidence_log_lr(x) for x in others], trim_fraction
                )
                alt = m_eff_alt * mean_alt
            log_odds_without = log_odds - contrib[cid] + alt
            p_without = sigmoid(log_odds_without)
            influence.append(InfluenceItem(
                evidence_id=ev.id,
                claim=ev.claim,
                polarity=ev.polarity,
                delta_pp=abs(p_neutral - p_without),
            ))

    return p_neutral, meta, influence


def blend_market(p_neutral: float, market_prob: float, alpha: float = MARKET_BLEND_ALPHA) -> float:
    return sigmoid(logit(p_neutral) + alpha * logit(market_prob))


# ---------------------------------------------------------------------------
# Top-level convenience
# ---------------------------------------------------------------------------


def run_forecast(
    p0: float,
    evidence: list[Evidence],
    market_prob: float,
    rho_by_cluster: dict[str, float] | None = None,
    alpha: float = MARKET_BLEND_ALPHA,
) -> ForecastResult:
    """Run a full Bayesian aggregation + market-blend forecast."""
    p_neutral, clusters, influence = aggregate_neutral(p0, evidence, rho_by_cluster)
    p_blended = blend_market(p_neutral, market_prob, alpha)
    edge_pct = (p_blended - market_prob) * 100.0

    count_for = sum(1 for e in evidence if e.polarity > 0)
    count_against = sum(1 for e in evidence if e.polarity < 0)

    return ForecastResult(
        prior=p0,
        p_neutral=p_neutral,
        p_blended=p_blended,
        market_prob=market_prob,
        edge_percent=edge_pct,
        evidence=evidence,
        clusters=clusters,
        influence=influence,
        evidence_count_for=count_for,
        evidence_count_against=count_against,
    )
