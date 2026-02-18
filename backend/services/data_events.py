from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


class EventType:
    """Canonical event type string constants.

    Use these constants everywhere instead of bare string literals so that
    typos are caught at import time rather than silently failing at runtime::

        # Bad — typo silently fires to zero handlers:
        subscriptions = ["market_data_refesh"]

        # Good — typo raises ValueError when the class is defined:
        subscriptions = [EventType.MARKET_DATA_REFRESH]

    All values in ``_ALL`` are accepted by ``EventDispatcher.subscribe()``.
    """

    PRICE_CHANGE: str = "price_change"
    """A market's price changed (from WebSocket feed)."""

    MARKET_DATA_REFRESH: str = "market_data_refresh"
    """Periodic batch of all market data (replaces scanner poll)."""

    MARKET_RESOLVED: str = "market_resolved"
    """A market outcome was determined."""

    CRYPTO_UPDATE: str = "crypto_update"
    """Crypto market data from the crypto worker."""

    WEATHER_UPDATE: str = "weather_update"
    """Weather forecast data from the weather worker."""

    TRADER_ACTIVITY: str = "trader_activity"
    """Smart wallet / copy trading signal from the tracked traders worker."""

    NEWS_UPDATE: str = "news_update"
    """News intent signal from the news worker."""

    NEWS_EVENT: str = "news_event"
    """Breaking news signal from the news worker (alias for compatibility)."""

    WORLD_INTEL_UPDATE: str = "world_intel_update"
    """Geopolitical and world event signals from the world intelligence worker.

    payload = {
        "signals": [
            {
                "signal_id": str,           # stable hash ID: wi_{hash}
                "signal_type": str,         # conflict|tension|instability|military|
                                            #   infrastructure|anomaly|convergence|
                                            #   earthquake|news
                "severity": float,          # 0.0-1.0 normalized
                "country": str | None,      # ISO3 or "ISO3A-ISO3B" for tension pairs
                "latitude": float | None,
                "longitude": float | None,
                "title": str,
                "description": str,
                "source": str,              # acled|gdelt|military|usgs|rss_news|...
                "detected_at": str,         # ISO datetime UTC
                "related_market_ids": list[str],
                "market_relevance_score": float,  # 0.0-1.0
                "metadata": dict,           # signal-type-specific enrichment
            },
            ...
        ],
        "summary": {
            "total": int,
            "critical": int,               # signals with severity >= 0.7
            "by_type": dict[str, int],     # count per signal_type
            "collection_duration_seconds": float,
        },
    }
    """

    _ALL: frozenset[str] = frozenset(
        {
            "price_change",
            "market_data_refresh",
            "market_resolved",
            "crypto_update",
            "weather_update",
            "trader_activity",
            "news_update",
            "news_event",
            "world_intel_update",
        }
    )


@dataclass(frozen=True)
class DataEvent:
    """A single data event that strategies can subscribe to.

    Event types (use ``EventType.*`` constants):
        EventType.PRICE_CHANGE          - A market's price changed (from WS feed)
        EventType.MARKET_DATA_REFRESH   - Periodic batch of all market data
        EventType.MARKET_RESOLVED       - A market outcome was determined
        EventType.CRYPTO_UPDATE         - Crypto market data from crypto worker
        EventType.WEATHER_UPDATE        - Weather forecast data from weather worker
        EventType.TRADER_ACTIVITY       - Smart wallet / copy trading signal
        EventType.NEWS_UPDATE           - News intent signal from news worker
        EventType.NEWS_EVENT            - Breaking news signal from news worker
        EventType.WORLD_INTEL_UPDATE    - Geopolitical signals from world intel worker
    """

    event_type: str
    source: str
    timestamp: datetime
    market_id: Optional[str] = None
    token_id: Optional[str] = None
    payload: dict = field(default_factory=dict)
    old_price: Optional[float] = None
    new_price: Optional[float] = None
    markets: Optional[list] = None
    events: Optional[list] = None
    prices: Optional[dict] = None
    scan_mode: Optional[str] = None
    changed_token_ids: Optional[list[str]] = None
    changed_market_ids: Optional[list[str]] = None
    affected_market_ids: Optional[list[str]] = None
