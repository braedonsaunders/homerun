"""World Intelligence API routes."""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Query

router = APIRouter(tags=["world-intelligence"])
logger = logging.getLogger(__name__)


@router.get("/world-intelligence/signals")
async def get_world_signals(
    signal_type: Optional[str] = Query(None, description="Filter by signal type"),
    country: Optional[str] = Query(None, description="Filter by country"),
    min_severity: float = Query(0.0, description="Minimum severity (0-1)"),
    limit: int = Query(100, ge=1, le=500),
):
    """Get current world intelligence signals."""
    from services.world_intelligence import signal_aggregator

    signals = signal_aggregator._last_signals or []
    if signal_type:
        signals = [s for s in signals if s.signal_type == signal_type]
    if country:
        country_lower = country.lower()
        signals = [s for s in signals if s.country and s.country.lower() == country_lower]
    signals = [s for s in signals if s.severity >= min_severity]
    signals = sorted(signals, key=lambda s: s.severity, reverse=True)[:limit]

    return {
        "signals": [_signal_to_dict(s) for s in signals],
        "total": len(signals),
        "last_collection": signal_aggregator._last_collection_at.isoformat() if signal_aggregator._last_collection_at else None,
    }


@router.get("/world-intelligence/instability")
async def get_instability_scores(
    country: Optional[str] = Query(None),
    min_score: float = Query(0.0),
    limit: int = Query(50, ge=1, le=200),
):
    """Get country instability index scores."""
    from services.world_intelligence import instability_scorer

    scores = instability_scorer.get_all_scores()
    if country:
        scores = {k: v for k, v in scores.items() if k.lower() == country.lower() or (v.iso3 and v.iso3.lower() == country.lower())}
    else:
        scores = {k: v for k, v in scores.items() if v.score >= min_score}

    sorted_scores = sorted(scores.values(), key=lambda s: s.score, reverse=True)[:limit]
    return {
        "scores": [
            {
                "country": s.country,
                "iso3": s.iso3,
                "score": round(s.score, 1),
                "trend": s.trend,
                "change_24h": round(s.change_24h, 1) if s.change_24h else None,
                "change_7d": round(s.change_7d, 1) if s.change_7d else None,
                "components": s.components,
                "contributing_signals": s.contributing_signals[:5],
                "last_updated": s.last_updated.isoformat() if s.last_updated else None,
            }
            for s in sorted_scores
        ],
        "total": len(sorted_scores),
    }


@router.get("/world-intelligence/tensions")
async def get_tension_pairs(
    min_tension: float = Query(0.0),
    limit: int = Query(20, ge=1, le=100),
):
    """Get country-pair tension scores."""
    from services.world_intelligence import tension_tracker

    pairs = tension_tracker.get_all_tensions()
    pairs = [p for p in pairs if p.tension_score >= min_tension]
    pairs = sorted(pairs, key=lambda p: p.tension_score, reverse=True)[:limit]

    return {
        "tensions": [
            {
                "country_a": p.country_a,
                "country_b": p.country_b,
                "tension_score": round(p.tension_score, 1),
                "event_count": p.event_count,
                "avg_goldstein_scale": round(p.avg_goldstein_scale, 2) if p.avg_goldstein_scale else None,
                "trend": p.trend,
                "top_event_types": p.top_event_types,
                "last_updated": p.last_updated.isoformat() if p.last_updated else None,
            }
            for p in pairs
        ],
        "total": len(pairs),
    }


@router.get("/world-intelligence/convergences")
async def get_convergence_zones():
    """Get active geo-convergence zones where multiple signal types overlap."""
    from services.world_intelligence import convergence_detector

    zones = convergence_detector.get_active_convergences()
    return {
        "zones": [
            {
                "grid_key": z.grid_key,
                "latitude": z.latitude,
                "longitude": z.longitude,
                "signal_types": list(z.signal_types),
                "signal_count": z.signal_count,
                "urgency_score": round(z.urgency_score, 1),
                "country": z.country,
                "nearby_markets": z.nearby_markets,
                "detected_at": z.detected_at.isoformat() if z.detected_at else None,
            }
            for z in zones
        ],
        "total": len(zones),
    }


@router.get("/world-intelligence/anomalies")
async def get_temporal_anomalies(
    min_severity: str = Query("medium", description="Minimum severity: normal, medium, high, critical"),
):
    """Get temporal baseline anomalies (z-score deviations)."""
    from services.world_intelligence import anomaly_detector

    severity_order = {"normal": 0, "medium": 1, "high": 2, "critical": 3}
    min_level = severity_order.get(min_severity, 1)

    anomalies = anomaly_detector.detect_anomalies()
    anomalies = [a for a in anomalies if severity_order.get(a.severity, 0) >= min_level]
    anomalies = sorted(anomalies, key=lambda a: abs(a.z_score), reverse=True)

    return {
        "anomalies": [
            {
                "signal_type": a.signal_type,
                "country": a.country,
                "z_score": round(a.z_score, 2),
                "severity": a.severity,
                "current_value": a.current_value,
                "baseline_mean": round(a.baseline_mean, 2),
                "baseline_std": round(a.baseline_std, 2),
                "description": a.description,
                "detected_at": a.detected_at.isoformat() if a.detected_at else None,
            }
            for a in anomalies
        ],
        "total": len(anomalies),
    }


@router.get("/world-intelligence/military")
async def get_military_activity():
    """Get current military activity summary."""
    from services.world_intelligence import military_monitor

    summary = await military_monitor.get_activity_summary()
    return summary


@router.get("/world-intelligence/infrastructure")
async def get_infrastructure_events():
    """Get current infrastructure disruptions and cascade risks."""
    from services.world_intelligence import infrastructure_monitor

    disruptions = await infrastructure_monitor.get_current_disruptions()
    cascade_risks = infrastructure_monitor.get_cascade_risks()

    return {
        "disruptions": [
            {
                "event_type": e.event_type,
                "country": e.country,
                "severity": round(e.severity, 2),
                "started_at": e.started_at.isoformat() if e.started_at else None,
                "description": e.description,
                "source": e.source,
                "cascade_risk_score": round(e.cascade_risk_score, 2),
            }
            for e in disruptions
        ],
        "cascade_risks": cascade_risks,
        "total_disruptions": len(disruptions),
    }


@router.get("/world-intelligence/summary")
async def get_world_intelligence_summary():
    """Get a high-level summary of all world intelligence data."""
    from services.world_intelligence import (
        signal_aggregator,
        instability_scorer,
        tension_tracker,
        anomaly_detector,
        convergence_detector,
    )

    signal_summary = signal_aggregator.get_signal_summary()
    critical_countries = instability_scorer.get_critical_countries()
    high_tensions = tension_tracker.get_high_tension_pairs()
    critical_anomalies = anomaly_detector.get_critical_anomalies()
    convergences = convergence_detector.get_active_convergences()

    return {
        "signal_summary": signal_summary,
        "critical_countries": [
            {"country": c.country, "iso3": c.iso3, "score": round(c.score, 1), "trend": c.trend}
            for c in critical_countries[:10]
        ],
        "high_tensions": [
            {"pair": f"{p.country_a}-{p.country_b}", "score": round(p.tension_score, 1), "trend": p.trend}
            for p in high_tensions[:10]
        ],
        "critical_anomalies": len(critical_anomalies),
        "active_convergences": len(convergences),
        "last_collection": signal_aggregator._last_collection_at.isoformat() if signal_aggregator._last_collection_at else None,
    }


@router.get("/world-intelligence/status")
async def get_world_intelligence_status():
    """Get world intelligence worker status."""
    from models.database import AsyncSessionLocal, WorldIntelligenceSnapshot
    from sqlalchemy import select

    try:
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(WorldIntelligenceSnapshot).where(
                    WorldIntelligenceSnapshot.id == "latest"
                )
            )
            snapshot = result.scalar_one_or_none()
            if snapshot:
                return {
                    "status": snapshot.status,
                    "stats": snapshot.stats,
                    "updated_at": snapshot.updated_at.isoformat() if snapshot.updated_at else None,
                }
    except Exception as e:
        logger.warning("Failed to read world intelligence snapshot: %s", e)

    return {"status": {"running": False}, "stats": {}, "updated_at": None}


def _signal_to_dict(s) -> dict:
    """Convert a WorldSignal dataclass to a serializable dict."""
    return {
        "signal_id": s.signal_id,
        "signal_type": s.signal_type,
        "severity": round(s.severity, 3),
        "country": s.country,
        "latitude": s.latitude,
        "longitude": s.longitude,
        "title": s.title,
        "description": s.description,
        "source": s.source,
        "detected_at": s.detected_at.isoformat() if s.detected_at else None,
        "metadata": s.metadata,
        "related_market_ids": s.related_market_ids,
        "market_relevance_score": round(s.market_relevance_score, 3) if s.market_relevance_score else None,
    }
