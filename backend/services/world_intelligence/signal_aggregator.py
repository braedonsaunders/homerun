"""Master World Intelligence signal aggregator.

Orchestrates a full collection cycle across all world intelligence
sources (ACLED, GDELT, OpenSky, Cloudflare Radar, etc.) and normalises
every output into a unified ``WorldSignal`` stream.  The aggregated
feed is the primary interface that downstream prediction-market
strategies consume.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from config import settings

from .acled_client import acled_client, ConflictEvent
from .anomaly_detector import anomaly_detector, TemporalAnomaly
from .convergence_detector import convergence_detector, ConvergenceZone
from .infrastructure_monitor import infrastructure_monitor, InfrastructureEvent
from .instability_scorer import instability_scorer, CountryInstabilityScore
from .military_monitor import military_monitor, MilitaryActivity
from .tension_tracker import tension_tracker, CountryPairTension

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

_SIGNAL_TYPES = {
    "conflict",
    "tension",
    "instability",
    "convergence",
    "anomaly",
    "military",
    "infrastructure",
}

# Market relevance scoring weights
_RELEVANCE_DIRECT_COUNTRY = 0.9
_RELEVANCE_RELATED_COUNTRY = 0.6
_RELEVANCE_KEYWORD = 0.4
_RELEVANCE_REGION = 0.3


@dataclass
class WorldSignal:
    """A normalised signal from any world intelligence source."""

    signal_id: str
    signal_type: str  # one of _SIGNAL_TYPES
    severity: float  # 0-1 normalised
    country: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    title: str = ""
    description: str = ""
    source: str = ""
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)
    related_market_ids: list[str] = field(default_factory=list)
    market_relevance_score: float = 0.0


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------


def _signal_id() -> str:
    return str(uuid.uuid4())


def _conflict_to_signal(event: ConflictEvent) -> WorldSignal:
    """Convert a ConflictEvent to a WorldSignal."""
    from .acled_client import ACLEDClient

    severity = ACLEDClient.get_severity_score(event)
    return WorldSignal(
        signal_id=_signal_id(),
        signal_type="conflict",
        severity=severity,
        country=event.iso3,
        latitude=event.latitude,
        longitude=event.longitude,
        title=f"{event.event_type.title()} in {event.country}",
        description=(
            f"{event.sub_event_type} — {event.fatalities} fatalities. "
            f"{event.notes[:200]}" if event.notes else event.sub_event_type
        ),
        source="acled",
        metadata={
            "event_id": event.event_id,
            "event_type": event.event_type,
            "sub_event_type": event.sub_event_type,
            "fatalities": event.fatalities,
        },
    )


def _tension_to_signal(tension: CountryPairTension) -> WorldSignal:
    """Convert a CountryPairTension to a WorldSignal."""
    severity = tension.tension_score / 100.0
    return WorldSignal(
        signal_id=_signal_id(),
        signal_type="tension",
        severity=severity,
        country=f"{tension.country_a}-{tension.country_b}",
        title=f"Tension: {tension.country_a}-{tension.country_b} ({tension.tension_score:.0f})",
        description=(
            f"Trend: {tension.trend}, {tension.event_count} events, "
            f"Goldstein: {tension.avg_goldstein_scale:.1f}"
        ),
        source="gdelt",
        metadata={
            "country_a": tension.country_a,
            "country_b": tension.country_b,
            "trend": tension.trend,
            "event_count": tension.event_count,
            "top_event_types": tension.top_event_types,
        },
    )


def _instability_to_signal(score: CountryInstabilityScore) -> WorldSignal:
    """Convert a CountryInstabilityScore to a WorldSignal."""
    severity = score.score / 100.0
    return WorldSignal(
        signal_id=_signal_id(),
        signal_type="instability",
        severity=severity,
        country=score.iso3,
        title=f"Instability: {score.iso3} ({score.score:.0f}/100)",
        description=(
            f"Trend: {score.trend}, "
            f"24h change: {score.change_24h:+.1f}, "
            f"7d change: {score.change_7d:+.1f}"
        ),
        source="cii",
        metadata={
            "components": score.components,
            "contributing_signals": score.contributing_signals,
        },
    )


def _convergence_to_signal(zone: ConvergenceZone) -> WorldSignal:
    """Convert a ConvergenceZone to a WorldSignal."""
    severity = zone.urgency_score / 100.0
    return WorldSignal(
        signal_id=_signal_id(),
        signal_type="convergence",
        severity=severity,
        country=zone.country,
        latitude=zone.latitude,
        longitude=zone.longitude,
        title=f"Convergence: {len(zone.signal_types)} types at {zone.grid_key}",
        description=(
            f"Signals: {', '.join(sorted(zone.signal_types))}, "
            f"Count: {zone.signal_count}, Country: {zone.country}"
        ),
        source="convergence_detector",
        metadata={
            "grid_key": zone.grid_key,
            "signal_types": sorted(zone.signal_types),
            "signal_count": zone.signal_count,
        },
    )


def _anomaly_to_signal(anomaly: TemporalAnomaly) -> WorldSignal:
    """Convert a TemporalAnomaly to a WorldSignal."""
    # Map severity labels to numeric values
    severity_map = {"normal": 0.2, "medium": 0.5, "high": 0.7, "critical": 0.9}
    severity = severity_map.get(anomaly.severity, 0.3)
    return WorldSignal(
        signal_id=_signal_id(),
        signal_type="anomaly",
        severity=severity,
        country=anomaly.country,
        title=f"Anomaly: {anomaly.signal_type} in {anomaly.country} (z={anomaly.z_score:.1f})",
        description=anomaly.description,
        source="anomaly_detector",
        metadata={
            "z_score": anomaly.z_score,
            "current_value": anomaly.current_value,
            "baseline_mean": anomaly.baseline_mean,
            "baseline_std": anomaly.baseline_std,
        },
    )


def _military_to_signal(activity: MilitaryActivity) -> WorldSignal:
    """Convert a MilitaryActivity to a WorldSignal."""
    severity = 0.5 if not activity.is_unusual else 0.7
    return WorldSignal(
        signal_id=_signal_id(),
        signal_type="military",
        severity=severity,
        country=activity.country,
        latitude=activity.latitude,
        longitude=activity.longitude,
        title=f"Military {activity.activity_type}: {activity.callsign} ({activity.region})",
        description=(
            f"{activity.aircraft_type}, alt={activity.altitude:.0f}m, "
            f"speed={activity.speed:.0f}m/s, heading={activity.heading:.0f}deg"
        ),
        source="opensky",
        metadata={
            "callsign": activity.callsign,
            "aircraft_type": activity.aircraft_type,
            "region": activity.region,
            "is_unusual": activity.is_unusual,
        },
    )


def _infrastructure_to_signal(event: InfrastructureEvent) -> WorldSignal:
    """Convert an InfrastructureEvent to a WorldSignal."""
    return WorldSignal(
        signal_id=_signal_id(),
        signal_type="infrastructure",
        severity=event.severity,
        country=event.country,
        title=f"Infrastructure: {event.event_type} in {event.country}",
        description=event.description,
        source=event.source,
        metadata={
            "event_type": event.event_type,
            "affected_services": event.affected_services,
            "cascade_risk_score": event.cascade_risk_score,
        },
    )


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


class WorldSignalAggregator:
    """Orchestrates collection from all world intelligence sources.

    A single call to ``run_collection_cycle`` fetches data from every
    registered source, feeds cross-cutting detectors (convergence,
    anomalies), and returns a unified, severity-sorted signal list.
    """

    def __init__(self) -> None:
        self._last_signals: list[WorldSignal] = []
        self._last_collection_at: Optional[datetime] = None

    # -- Collection orchestration --------------------------------------------

    async def run_collection_cycle(self) -> list[WorldSignal]:
        """Execute one full collection cycle across all sources.

        Steps:
            1. Fetch ACLED conflict events
            2. Update tension tracker
            3. Fetch military activity
            4. Fetch infrastructure disruptions
            5. Feed all signals into convergence detector
            6. Run anomaly detection
            7. Compute instability scores
            8. Normalise all outputs into WorldSignal format
            9. Return unified signal list sorted by severity desc
        """
        signals: list[WorldSignal] = []

        # 1. ACLED conflict events
        conflict_events: list[ConflictEvent] = []
        try:
            conflict_events = await acled_client.fetch_recent(hours=24)
            for ev in conflict_events:
                signals.append(_conflict_to_signal(ev))
        except Exception as exc:
            logger.error("ACLED collection failed: %s", exc)

        # 2. Tension tracker
        tensions: list[CountryPairTension] = []
        try:
            tensions = await tension_tracker.update_tensions()
            for t in tensions:
                signals.append(_tension_to_signal(t))
        except Exception as exc:
            logger.error("Tension tracker collection failed: %s", exc)

        # 3. Military activity
        military_events: list[MilitaryActivity] = []
        try:
            military_events = await military_monitor.fetch_military_flights()
            for m in military_events:
                signals.append(_military_to_signal(m))
        except Exception as exc:
            logger.error("Military monitor collection failed: %s", exc)

        # 4. Infrastructure disruptions
        infra_events: list[InfrastructureEvent] = []
        try:
            infra_events = await infrastructure_monitor.get_current_disruptions()
            for ie in infra_events:
                signals.append(_infrastructure_to_signal(ie))
        except Exception as exc:
            logger.error("Infrastructure monitor collection failed: %s", exc)

        # 5. Feed signals into convergence detector
        try:
            for ev in conflict_events:
                sig_type = "protest" if ev.event_type in ("protests", "riots") else "conflict"
                convergence_detector.ingest_signal(
                    sig_type, ev.latitude, ev.longitude,
                    metadata={"country": ev.iso3, "severity": acled_client.get_severity_score(ev)},
                )
            for m in military_events:
                sig_type = "military_flight" if m.activity_type == "flight" else "military_vessel"
                convergence_detector.ingest_signal(
                    sig_type, m.latitude, m.longitude,
                    metadata={"country": m.country, "region": m.region},
                )
            for ie in infra_events:
                if ie.event_type == "internet_outage":
                    # Internet outages don't always have coords; skip if missing
                    convergence_detector.ingest_signal(
                        "internet_outage", 0.0, 0.0,
                        metadata={"country": ie.country, "severity": ie.severity},
                    )

            convergences = convergence_detector.detect_convergences(min_types=3)
            for cz in convergences:
                signals.append(_convergence_to_signal(cz))
        except Exception as exc:
            logger.error("Convergence detection failed: %s", exc)

        # 6. Anomaly detection - record observations and detect
        try:
            # Record conflict event counts per country
            conflict_counts = acled_client.get_country_event_counts(conflict_events)
            for iso3, type_counts in conflict_counts.items():
                total = sum(type_counts.values())
                anomaly_detector.record_observation("conflict_events", iso3, float(total))
                protest_count = type_counts.get("protests", 0) + type_counts.get("riots", 0)
                if protest_count > 0:
                    anomaly_detector.record_observation("protests", iso3, float(protest_count))

            # Record military flight counts per region
            from collections import defaultdict as _dd
            region_counts: dict[str, int] = _dd(int)
            for m in military_events:
                region_counts[m.region] += 1
            for region, count in region_counts.items():
                anomaly_detector.record_observation("military_flights", region, float(count))

            anomalies = anomaly_detector.detect_anomalies()
            for a in anomalies:
                signals.append(_anomaly_to_signal(a))
        except Exception as exc:
            logger.error("Anomaly detection failed: %s", exc)

        # 7. Instability scores
        try:
            scores = await instability_scorer.compute_scores(
                conflict_events=conflict_events,
                military_events=military_events,
                news_velocity={},  # Would be populated by news service
                protest_events=[e for e in conflict_events if e.event_type in ("protests", "riots")],
            )
            for score in scores.values():
                # Only emit signals for countries with meaningful scores
                if score.score >= 30:
                    signals.append(_instability_to_signal(score))
        except Exception as exc:
            logger.error("Instability scoring failed: %s", exc)

        # 8. Sort by severity descending
        signals.sort(key=lambda s: s.severity, reverse=True)

        self._last_signals = signals
        self._last_collection_at = datetime.now(timezone.utc)

        logger.info(
            "Collection cycle complete: %d signals (%d critical)",
            len(signals),
            sum(1 for s in signals if s.severity >= 0.7),
        )
        return signals

    # -- Market matching -----------------------------------------------------

    async def match_signals_to_markets(
        self,
        signals: list[WorldSignal],
        active_markets: list[Any],
    ) -> list[WorldSignal]:
        """Enrich signals with related_market_ids and relevance scores.

        Matching strategies (cumulative, takes highest score):
        - direct_country_match: market question contains the signal country -> 0.9
        - related_country_match: market question contains a tension partner -> 0.6
        - keyword_match: market question contains signal-type keywords -> 0.4
        - region_match: market question references the same region -> 0.3
        """
        _KEYWORD_MAP: dict[str, list[str]] = {
            "conflict": ["war", "conflict", "attack", "ceasefire", "invasion"],
            "tension": ["sanctions", "diplomacy", "tensions", "relations"],
            "instability": ["coup", "revolution", "collapse", "crisis"],
            "military": ["military", "defense", "nato", "nuclear", "missile"],
            "infrastructure": ["internet", "outage", "pipeline", "trade", "shipping"],
            "anomaly": ["surge", "spike", "unusual"],
            "convergence": ["escalation", "crisis"],
        }

        for signal in signals:
            best_relevance = 0.0
            matched_ids: list[str] = []

            for market in active_markets:
                market_id = str(getattr(market, "market_id", getattr(market, "id", "")))
                question = str(
                    getattr(market, "question", getattr(market, "title", ""))
                ).lower()

                relevance = 0.0

                # Direct country match
                if signal.country and signal.country.lower() in question:
                    relevance = max(relevance, _RELEVANCE_DIRECT_COUNTRY)

                # Related country match (for tension signals, check both countries)
                meta = signal.metadata
                for key in ("country_a", "country_b"):
                    related = meta.get(key, "")
                    if related and related.lower() in question:
                        relevance = max(relevance, _RELEVANCE_RELATED_COUNTRY)

                # Keyword match
                keywords = _KEYWORD_MAP.get(signal.signal_type, [])
                for kw in keywords:
                    if kw in question:
                        relevance = max(relevance, _RELEVANCE_KEYWORD)
                        break

                # Region match
                region = meta.get("region", "")
                if region and region.replace("_", " ") in question:
                    relevance = max(relevance, _RELEVANCE_REGION)

                if relevance > 0:
                    matched_ids.append(market_id)
                    best_relevance = max(best_relevance, relevance)

            signal.related_market_ids = matched_ids
            signal.market_relevance_score = round(best_relevance, 2)

        return signals

    # -- Accessors -----------------------------------------------------------

    def get_signal_summary(self) -> dict:
        """Overall signal counts by type and severity tier."""
        by_type: dict[str, int] = {}
        by_severity: dict[str, int] = {"low": 0, "medium": 0, "high": 0, "critical": 0}

        for s in self._last_signals:
            by_type[s.signal_type] = by_type.get(s.signal_type, 0) + 1
            if s.severity >= 0.8:
                by_severity["critical"] += 1
            elif s.severity >= 0.6:
                by_severity["high"] += 1
            elif s.severity >= 0.3:
                by_severity["medium"] += 1
            else:
                by_severity["low"] += 1

        return {
            "total": len(self._last_signals),
            "by_type": by_type,
            "by_severity": by_severity,
            "last_collection_at": (
                self._last_collection_at.isoformat()
                if self._last_collection_at
                else None
            ),
        }

    def get_critical_signals(self) -> list[WorldSignal]:
        """Return signals with severity >= 0.7."""
        return [s for s in self._last_signals if s.severity >= 0.7]

    def get_signals_for_country(self, country: str) -> list[WorldSignal]:
        """Return all signals associated with a country (case-insensitive)."""
        country_lower = country.lower()
        results: list[WorldSignal] = []
        for s in self._last_signals:
            if s.country and country_lower in s.country.lower():
                results.append(s)
                continue
            # Also check tension pairs and metadata
            for key in ("country_a", "country_b"):
                if country_lower in str(s.metadata.get(key, "")).lower():
                    results.append(s)
                    break
        return results

    def get_signals_for_market(self, market_id: str) -> list[WorldSignal]:
        """Return signals whose related_market_ids contain the given ID."""
        return [
            s for s in self._last_signals
            if market_id in s.related_market_ids
        ]


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

signal_aggregator = WorldSignalAggregator()
