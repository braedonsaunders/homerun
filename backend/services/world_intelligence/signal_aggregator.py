"""Master World Intelligence signal aggregator.

Orchestrates a full collection cycle across all world intelligence
sources (ACLED, GDELT, OpenSky, Cloudflare Radar, etc.) and normalises
every output into a unified ``WorldSignal`` stream.  The aggregated
feed is the primary interface that downstream prediction-market
strategies consume.
"""

from __future__ import annotations

import logging
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Optional

from config import settings

from .acled_client import acled_client, ConflictEvent
from .anomaly_detector import anomaly_detector, TemporalAnomaly
from .convergence_detector import convergence_detector, ConvergenceZone
from .infrastructure_monitor import infrastructure_monitor, InfrastructureEvent
from .instability_scorer import instability_scorer, CountryInstabilityScore
from .military_monitor import military_monitor, MilitaryActivity
from .chokepoint_feed import chokepoint_feed
from .military_catalog import military_catalog
from .tension_tracker import tension_tracker, CountryPairTension
from .usgs_client import usgs_client, Earthquake
from .taxonomy_catalog import taxonomy_catalog
from services.news.gov_rss_feeds import gov_rss_service, GovArticle

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

_SIGNAL_TYPES = {
    *taxonomy_catalog.world_signal_types(),
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


def _stable_signal_id(*parts: Any) -> str:
    packed = "|".join(str(p or "") for p in parts)
    digest = hashlib.sha256(packed.encode("utf-8")).hexdigest()[:20]
    return f"wi_{digest}"


def _normalize_iso3(value: str | None) -> str:
    text = str(value or "").strip().upper()
    if not text:
        return ""
    if len(text) == 3 and text.isalpha():
        return text
    return military_catalog.country_aliases().get(text, "")


def _conflict_to_signal(event: ConflictEvent) -> WorldSignal:
    """Convert a ConflictEvent to a WorldSignal."""
    from .acled_client import ACLEDClient

    severity = ACLEDClient.get_severity_score(event)
    return WorldSignal(
        signal_id=_stable_signal_id("acled", event.event_id),
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
        signal_id=_stable_signal_id("gdelt", tension.country_a, tension.country_b),
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
        signal_id=_stable_signal_id("cii", score.iso3),
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
        signal_id=_stable_signal_id("conv", zone.grid_key),
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
        signal_id=_stable_signal_id("anomaly", anomaly.signal_type, anomaly.country),
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
    source = str(activity.provider or ("opensky" if activity.activity_type == "flight" else "aisstream"))
    providers = list(activity.providers or ([source] if source else []))
    return WorldSignal(
        signal_id=_stable_signal_id(
            source,
            activity.callsign,
            activity.transponder,
            activity.region,
            round(activity.latitude, 2),
            round(activity.longitude, 2),
        ),
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
        source=source,
        metadata={
            "callsign": activity.callsign,
            "transponder": activity.transponder,
            "aircraft_type": activity.aircraft_type,
            "region": activity.region,
            "is_unusual": activity.is_unusual,
            "activity_type": activity.activity_type,
            "provider": source,
            "providers": providers,
        },
    )


def _infrastructure_to_signal(event: InfrastructureEvent) -> WorldSignal:
    """Convert an InfrastructureEvent to a WorldSignal."""
    return WorldSignal(
        signal_id=_stable_signal_id(
            "infra",
            event.event_type,
            event.country,
            event.description[:80],
        ),
        signal_type="infrastructure",
        severity=event.severity,
        country=event.country,
        latitude=event.latitude,
        longitude=event.longitude,
        title=f"Infrastructure: {event.event_type} in {event.country}",
        description=event.description,
        source=event.source,
        metadata={
            "event_type": event.event_type,
            "affected_services": event.affected_services,
            "cascade_risk_score": event.cascade_risk_score,
        },
    )


def _gov_article_to_signal(article: GovArticle) -> WorldSignal:
    severity_map = {
        "critical": 0.75,
        "high": 0.6,
        "medium": 0.4,
    }
    severity = severity_map.get(article.priority, 0.35)
    return WorldSignal(
        signal_id=_stable_signal_id("gov_rss", article.article_id),
        signal_type="tension",
        severity=severity,
        country=_normalize_iso3(article.country_iso3),
        title=f"{article.source}: {article.title}",
        description=article.summary or article.title,
        source="gov_rss",
        detected_at=article.published or article.fetched_at,
        metadata={
            "agency": article.agency,
            "priority": article.priority,
            "url": article.url,
        },
    )


def _earthquake_to_signal(quake: Earthquake) -> WorldSignal:
    return WorldSignal(
        signal_id=_stable_signal_id("usgs", quake.event_id),
        signal_type="anomaly",
        severity=max(0.15, min(1.0, float(quake.severity_score or 0.0))),
        country=_normalize_iso3(quake.country) or None,
        latitude=quake.latitude,
        longitude=quake.longitude,
        title=f"Earthquake M{quake.magnitude:.1f}: {quake.place}",
        description=(
            f"USGS significance {quake.significance}, depth {quake.depth_km:.1f} km"
            + (" (tsunami warning)" if quake.tsunami else "")
        ),
        source="usgs",
        detected_at=quake.timestamp,
        metadata={
            "magnitude": quake.magnitude,
            "depth_km": quake.depth_km,
            "tsunami": quake.tsunami,
            "alert": quake.alert,
            "url": quake.url,
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
        self._last_source_status: dict[str, dict[str, Any]] = {}
        self._last_errors: list[str] = []

    async def _load_active_markets(self) -> list[Any]:
        try:
            from services.market_cache import market_cache_service

            if not market_cache_service._loaded:
                await market_cache_service.load_from_db()
            rows = getattr(market_cache_service, "_market_cache", {}) or {}
            markets: list[Any] = []
            for condition_id, raw in rows.items():
                if not isinstance(raw, dict):
                    continue
                if raw.get("active") is False:
                    continue
                question = str(raw.get("question") or raw.get("groupItemTitle") or "").strip()
                if not question:
                    continue
                markets.append(
                    SimpleNamespace(
                        market_id=str(condition_id),
                        question=question,
                        title=str(raw.get("groupItemTitle") or question),
                        latitude=raw.get("latitude"),
                        longitude=raw.get("longitude"),
                    )
                )
            return markets
        except Exception as exc:
            logger.debug("Failed loading active markets for world matching: %s", exc)
            return []

    # -- Collection orchestration --------------------------------------------

    async def run_collection_cycle(self) -> list[WorldSignal]:
        """Execute one full collection cycle across all sources.

        Steps:
            1. Fetch ACLED conflict events
            2. Update tension tracker
            3. Fetch military activity (multi-provider + dedupe)
            4. Fetch infrastructure disruptions
            5. Refresh maintained chokepoint feed cache
            6. Feed all signals into convergence detector
            7. Run anomaly detection
            8. Compute instability scores
            9. Normalise all outputs into WorldSignal format
            10. Return unified signal list sorted by severity desc
        """
        datetime.now(timezone.utc)
        signals: list[WorldSignal] = []
        source_status: dict[str, dict[str, Any]] = {}
        errors: list[str] = []

        def _record_source(
            name: str,
            started_at: datetime,
            count: int,
            error: Optional[Exception] = None,
            ok: Optional[bool] = None,
            error_message: Optional[str] = None,
            extra: Optional[dict[str, Any]] = None,
        ) -> None:
            elapsed = (datetime.now(timezone.utc) - started_at).total_seconds()
            source_ok = (error is None) if ok is None else bool(ok)
            reason = error_message or (str(error) if error else None)
            payload = {
                "ok": source_ok,
                "count": int(count),
                "duration_seconds": round(elapsed, 3),
                "error": reason,
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
            if extra:
                payload.update(extra)
            source_status[name] = payload
            if not source_ok and reason:
                errors.append(f"{name}: {reason}")

        # 1. ACLED conflict events
        conflict_events: list[ConflictEvent] = []
        started = datetime.now(timezone.utc)
        try:
            conflict_events = await acled_client.fetch_recent(hours=24)
            for ev in conflict_events:
                signals.append(_conflict_to_signal(ev))
            health = acled_client.get_health()
            acled_error = str(health.get("last_error") or "").strip() or None
            _record_source(
                "acled",
                started,
                len(conflict_events),
                ok=not acled_error,
                error_message=acled_error,
                extra=health,
            )
        except Exception as exc:
            logger.error("ACLED collection failed: %s", exc)
            _record_source("acled", started, 0, error=exc, extra=acled_client.get_health())

        # 2. Tension tracker
        tensions: list[CountryPairTension] = []
        started = datetime.now(timezone.utc)
        try:
            tensions = await tension_tracker.update_tensions()
            for t in tensions:
                signals.append(_tension_to_signal(t))
            tension_health = tension_tracker.get_health()
            tension_error = str(tension_health.get("last_error") or "").strip() or None
            _record_source(
                "gdelt_tensions",
                started,
                len(tensions),
                ok=not tension_error,
                error_message=tension_error,
                extra=tension_health,
            )
        except Exception as exc:
            logger.error("Tension tracker collection failed: %s", exc)
            _record_source(
                "gdelt_tensions",
                started,
                0,
                error=exc,
                extra=tension_tracker.get_health(),
            )

        # 3. Military activity
        military_events: list[MilitaryActivity] = []
        started = datetime.now(timezone.utc)
        try:
            military_enabled = bool(getattr(settings, "WORLD_INTEL_MILITARY_ENABLED", True))
            if military_enabled:
                flights = await military_monitor.fetch_military_flights()
                vessels = await military_monitor.fetch_vessel_activity()
                military_events = flights + vessels
                for m in military_events:
                    signals.append(_military_to_signal(m))
            military_health = military_monitor.get_health()
            military_error = str(
                military_health.get("last_error")
                or military_health.get("last_vessel_error")
                or ""
            ).strip() or None
            military_ok = (not military_error) or (not military_enabled)
            _record_source(
                "military",
                started,
                len(military_events),
                ok=military_ok,
                error_message=military_error,
                extra=military_health,
            )
        except Exception as exc:
            logger.error("Military monitor collection failed: %s", exc)
            _record_source(
                "military",
                started,
                0,
                error=exc,
                extra=military_monitor.get_health(),
            )

        # 4. Infrastructure disruptions
        infra_events: list[InfrastructureEvent] = []
        started = datetime.now(timezone.utc)
        try:
            infra_events = await infrastructure_monitor.get_current_disruptions()
            for ie in infra_events:
                signals.append(_infrastructure_to_signal(ie))
            infra_health = infrastructure_monitor.get_health()
            infra_error = str(infra_health.get("last_error") or "").strip() or None
            _record_source(
                "infrastructure",
                started,
                len(infra_events),
                ok=not infra_error,
                error_message=infra_error,
                extra=infra_health,
            )
        except Exception as exc:
            logger.error("Infrastructure monitor collection failed: %s", exc)
            _record_source("infrastructure", started, 0, error=exc)

        # 5. Government RSS
        gov_articles: list[GovArticle] = []
        started = datetime.now(timezone.utc)
        try:
            gov_articles = await gov_rss_service.fetch_all(consumer="world_intelligence")
            for article in gov_articles:
                signals.append(_gov_article_to_signal(article))
            gov_health = gov_rss_service.get_health()
            gov_error = str(gov_health.get("last_error") or "").strip() or None
            gov_ok = (not gov_error) or gov_error == "disabled"
            _record_source(
                "gov_rss",
                started,
                len(gov_articles),
                ok=gov_ok,
                error_message=gov_error,
                extra=gov_health,
            )
        except Exception as exc:
            logger.error("Gov RSS collection failed: %s", exc)
            _record_source("gov_rss", started, 0, error=exc)

        # 6. USGS earthquakes
        earthquakes: list[Earthquake] = []
        started = datetime.now(timezone.utc)
        try:
            usgs_enabled = bool(getattr(settings, "WORLD_INTEL_USGS_ENABLED", True))
            if usgs_enabled:
                earthquakes = await usgs_client.fetch_earthquakes(
                    feed="m4.5_day",
                    min_magnitude=float(
                        getattr(settings, "WORLD_INTEL_USGS_MIN_MAGNITUDE", 4.5) or 4.5
                    ),
                )
                for quake in earthquakes:
                    signals.append(_earthquake_to_signal(quake))
            usgs_health = usgs_client.get_health()
            usgs_error = str(usgs_health.get("last_error") or "").strip() or None
            usgs_ok = (not usgs_error) or (not usgs_enabled) or usgs_error == "disabled"
            _record_source(
                "usgs",
                started,
                len(earthquakes),
                ok=usgs_ok,
                error_message=usgs_error,
                extra=usgs_health,
            )
        except Exception as exc:
            logger.error("USGS collection failed: %s", exc)
            _record_source("usgs", started, 0, error=exc)

        # 7. Chokepoint feed refresh (no direct signal emission).
        started = datetime.now(timezone.utc)
        try:
            chokepoint_rows = await chokepoint_feed.refresh(force=False)
            chokepoint_health = chokepoint_feed.get_health()
            _record_source(
                "chokepoints",
                started,
                len(chokepoint_rows),
                ok=bool(chokepoint_health.get("ok")),
                error_message=str(chokepoint_health.get("last_error") or "").strip() or None,
                extra=chokepoint_health,
            )
        except Exception as exc:
            logger.error("Chokepoint feed refresh failed: %s", exc)
            _record_source(
                "chokepoints",
                started,
                0,
                error=exc,
                extra=chokepoint_feed.get_health(),
            )

        # 8. Convergence detection
        started = datetime.now(timezone.utc)
        try:
            signal_map = taxonomy_catalog.convergence_signal_map()
            for ev in conflict_events:
                sig_type = signal_map.get(ev.event_type, "conflict")
                convergence_detector.ingest_signal(
                    sig_type,
                    ev.latitude,
                    ev.longitude,
                    metadata={
                        "country": ev.iso3,
                        "severity": acled_client.get_severity_score(ev),
                    },
                )
            for m in military_events:
                sig_type = "military_flight" if m.activity_type == "flight" else "military_vessel"
                convergence_detector.ingest_signal(
                    sig_type,
                    m.latitude,
                    m.longitude,
                    metadata={"country": m.country, "region": m.region},
                )
            for quake in earthquakes:
                convergence_detector.ingest_signal(
                    "earthquake",
                    quake.latitude,
                    quake.longitude,
                    metadata={
                        "country": quake.country,
                        "severity": float(quake.severity_score or 0.0),
                    },
                )
            for ie in infra_events:
                lat = getattr(ie, "latitude", None)
                lon = getattr(ie, "longitude", None)
                if lat is None or lon is None:
                    continue
                convergence_detector.ingest_signal(
                    "internet_outage" if ie.event_type == "internet_outage" else "infrastructure_disruption",
                    float(lat),
                    float(lon),
                    metadata={
                        "country": _normalize_iso3(ie.country),
                        "severity": float(ie.severity or 0.0),
                    },
                )

            min_types = int(
                max(2, getattr(settings, "WORLD_INTEL_CONVERGENCE_MIN_TYPES", 2) or 2)
            )
            convergences = convergence_detector.detect_convergences(min_types=min_types)
            for cz in convergences:
                signals.append(_convergence_to_signal(cz))
            _record_source(
                "convergence",
                started,
                len(convergences),
                extra={"min_types": min_types},
            )
        except Exception as exc:
            logger.error("Convergence detection failed: %s", exc)
            _record_source("convergence", started, 0, error=exc)

        # 9. Anomaly detection
        started = datetime.now(timezone.utc)
        try:
            conflict_counts = acled_client.get_country_event_counts(conflict_events)
            for iso3, type_counts in conflict_counts.items():
                total = sum(type_counts.values())
                anomaly_detector.record_observation("conflict_events", iso3, float(total))
                protest_count = type_counts.get("protests", 0) + type_counts.get("riots", 0)
                if protest_count > 0:
                    anomaly_detector.record_observation("protests", iso3, float(protest_count))

            military_by_country: dict[str, dict[str, int]] = {}
            for m in military_events:
                iso3 = _normalize_iso3(m.country)
                if not iso3:
                    continue
                country_counts = military_by_country.setdefault(iso3, {})
                country_counts[m.activity_type] = country_counts.get(m.activity_type, 0) + 1
            for iso3, counts in military_by_country.items():
                flights = counts.get("flight", 0)
                vessels = counts.get("vessel", 0)
                if flights:
                    anomaly_detector.record_observation("military_flights", iso3, float(flights))
                if vessels:
                    anomaly_detector.record_observation("naval_vessels", iso3, float(vessels))

            outage_counts: dict[str, int] = {}
            for ie in infra_events:
                iso3 = _normalize_iso3(ie.country)
                if not iso3:
                    continue
                outage_counts[iso3] = outage_counts.get(iso3, 0) + 1
            for iso3, count in outage_counts.items():
                anomaly_detector.record_observation("internet_outages", iso3, float(count))

            news_counts: dict[str, int] = {}
            for article in gov_articles:
                iso3 = _normalize_iso3(article.country_iso3)
                if not iso3:
                    continue
                news_counts[iso3] = news_counts.get(iso3, 0) + 1
            for iso3, count in news_counts.items():
                anomaly_detector.record_observation("news_velocity", iso3, float(count))

            anomalies = anomaly_detector.detect_anomalies()
            for a in anomalies:
                signals.append(_anomaly_to_signal(a))
            _record_source("anomaly", started, len(anomalies))
        except Exception as exc:
            logger.error("Anomaly detection failed: %s", exc)
            _record_source("anomaly", started, 0, error=exc)

        # 10. Instability scores
        started = datetime.now(timezone.utc)
        try:
            news_velocity: dict[str, float] = {}
            for article in gov_articles:
                iso3 = _normalize_iso3(article.country_iso3)
                if not iso3:
                    continue
                news_velocity[iso3] = news_velocity.get(iso3, 0.0) + 1.0
            scores = await instability_scorer.compute_scores(
                conflict_events=conflict_events,
                military_events=military_events,
                news_velocity=news_velocity,
                protest_events=[e for e in conflict_events if e.event_type in ("protests", "riots")],
            )
            min_instability_signal = float(
                max(0.0, getattr(settings, "WORLD_INTEL_INSTABILITY_SIGNAL_MIN", 15.0) or 15.0)
            )
            emitted_instability = 0
            for score in scores.values():
                if score.score >= min_instability_signal:
                    signals.append(_instability_to_signal(score))
                    emitted_instability += 1
            _record_source(
                "instability",
                started,
                emitted_instability,
                extra={
                    "countries_scored": len(scores),
                    "min_signal_score": min_instability_signal,
                },
            )
        except Exception as exc:
            logger.error("Instability scoring failed: %s", exc)
            _record_source("instability", started, 0, error=exc)

        # 11. Market relevance matching (DB cache only; no trader orchestrator coupling)
        started = datetime.now(timezone.utc)
        try:
            active_markets = await self._load_active_markets()
            if active_markets:
                await self.match_signals_to_markets(signals, active_markets)
                matched = sum(1 for s in signals if s.related_market_ids)
                _record_source(
                    "market_matching",
                    started,
                    matched,
                    extra={"active_markets": len(active_markets)},
                )
            else:
                _record_source(
                    "market_matching",
                    started,
                    0,
                    extra={"active_markets": 0},
                )
        except Exception as exc:
            logger.error("World market-matching failed: %s", exc)
            _record_source("market_matching", started, 0, error=exc)

        # 12. Sort by severity descending
        signals.sort(key=lambda s: s.severity, reverse=True)

        self._last_signals = signals
        self._last_collection_at = datetime.now(timezone.utc)
        self._last_source_status = source_status
        self._last_errors = errors

        logger.info(
            "Collection cycle complete: %d signals (%d critical, %d source errors)",
            len(signals),
            sum(1 for s in signals if s.severity >= 0.7),
            len(errors),
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
        keyword_map = taxonomy_catalog.market_keyword_map()

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
                keywords = keyword_map.get(signal.signal_type, [])
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

    def export_runtime_state(self) -> dict[str, Any]:
        return {
            "anomaly_detector": anomaly_detector.export_state(),
            "convergence_detector": convergence_detector.export_state(),
            "military_monitor": military_monitor.export_state(),
        }

    def import_runtime_state(self, payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            return
        anomaly_detector.import_state(payload.get("anomaly_detector") or {})
        convergence_detector.import_state(payload.get("convergence_detector") or {})
        military_monitor.import_state(payload.get("military_monitor") or {})

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

    def get_source_status(self) -> dict[str, dict[str, Any]]:
        """Return per-source collection health/status from the last cycle."""
        return dict(self._last_source_status)

    def get_last_errors(self) -> list[str]:
        return list(self._last_errors)

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
