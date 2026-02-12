"""World Intelligence Signal Engine.

Aggregates non-market signals from conflict, military, infrastructure,
and geopolitical sources to inform prediction market trading.
"""

from .acled_client import acled_client, ACLEDClient, ConflictEvent
from .tension_tracker import tension_tracker, TensionTracker, CountryPairTension
from .instability_scorer import instability_scorer, InstabilityScorer, CountryInstabilityScore
from .convergence_detector import convergence_detector, ConvergenceDetector, ConvergenceZone
from .anomaly_detector import anomaly_detector, AnomalyDetector, TemporalAnomaly
from .military_monitor import military_monitor, MilitaryMonitor, MilitaryActivity
from .infrastructure_monitor import infrastructure_monitor, InfrastructureMonitor, InfrastructureEvent
from .signal_aggregator import signal_aggregator, WorldSignalAggregator, WorldSignal
from .signal_emitter import emit_world_intelligence_signals
from .resolver import (
    estimate_edge_percent,
    infer_direction,
    map_signal_to_strategy,
    resolve_world_signal_opportunities,
)
from .gov_rss_feeds import gov_rss_service, GovRSSFeedService
from .usgs_client import usgs_client, USGSClient, Earthquake
from .gdelt_events import gdelt_event_service, GDELTEventService
from .chokepoint_feed import chokepoint_feed, ChokepointFeed
from .region_catalog import region_catalog, RegionCatalog, Hotspot, Chokepoint
from .taxonomy_catalog import taxonomy_catalog, TaxonomyCatalog
from .military_catalog import military_catalog, MilitaryCatalog
from .infrastructure_catalog import infrastructure_catalog, InfrastructureCatalog
from .instability_catalog import instability_catalog, InstabilityCatalog
from .tension_pair_catalog import tension_pair_catalog, TensionPairCatalog
from .country_catalog import country_catalog, CountryCatalog

__all__ = [
    "acled_client", "ACLEDClient", "ConflictEvent",
    "tension_tracker", "TensionTracker", "CountryPairTension",
    "instability_scorer", "InstabilityScorer", "CountryInstabilityScore",
    "convergence_detector", "ConvergenceDetector", "ConvergenceZone",
    "anomaly_detector", "AnomalyDetector", "TemporalAnomaly",
    "military_monitor", "MilitaryMonitor", "MilitaryActivity",
    "infrastructure_monitor", "InfrastructureMonitor", "InfrastructureEvent",
    "signal_aggregator", "WorldSignalAggregator", "WorldSignal",
    "emit_world_intelligence_signals",
    "estimate_edge_percent",
    "infer_direction",
    "map_signal_to_strategy",
    "resolve_world_signal_opportunities",
    "gov_rss_service", "GovRSSFeedService",
    "usgs_client", "USGSClient", "Earthquake",
    "gdelt_event_service", "GDELTEventService",
    "chokepoint_feed", "ChokepointFeed",
    "region_catalog", "RegionCatalog", "Hotspot", "Chokepoint",
    "taxonomy_catalog", "TaxonomyCatalog",
    "military_catalog", "MilitaryCatalog",
    "infrastructure_catalog", "InfrastructureCatalog",
    "instability_catalog", "InstabilityCatalog",
    "tension_pair_catalog", "TensionPairCatalog",
    "country_catalog", "CountryCatalog",
]
