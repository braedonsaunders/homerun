from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
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

    DATA_SOURCE_UPDATE: str = "data_source_update"
    """A DB-defined data source completed a fetch cycle with new records.

    Strategies can subscribe to this event type to react when any data source
    produces new records, bridging the DataSource SDK into the real-time event
    pipeline.

    payload = {
        "source_slug": str,        # data source slug that produced the records
        "records_count": int,      # number of records upserted in this cycle
        "run_id": str,             # DataSourceRun row ID for audit trail
    }
    """

    EVENTS_UPDATE: str = "events_update"
    """Geopolitical and world event signals from the events worker.

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

    TRADE_EXECUTION: str = "trade_execution"
    """A trade was executed on the CLOB (from WebSocket trade tape).

    payload = {
        "price": float,       # execution price
        "size": float,        # size in shares
        "side": str,          # "BUY" or "SELL"
        "timestamp": float,   # epoch seconds
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
            "events_update",
            "data_source_update",
            "trade_execution",
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
        EventType.EVENTS_UPDATE    - Geopolitical signals from events worker
        EventType.DATA_SOURCE_UPDATE    - DB data source completed a fetch cycle
        EventType.TRADE_EXECUTION       - A trade executed on the CLOB
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

    def __post_init__(self) -> None:
        event_key = str(self.event_type or "").strip()
        if event_key not in EventType._ALL:
            raise ValueError(
                f"Unsupported event_type '{event_key}'. "
                f"Valid event types: {sorted(EventType._ALL)}"
            )
        if not isinstance(self.timestamp, datetime):
            raise TypeError("timestamp must be a datetime instance")
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware UTC")
        object.__setattr__(self, "event_type", event_key)
        object.__setattr__(self, "timestamp", self.timestamp.astimezone(timezone.utc))


class BlockReason:
    """Standard reason codes passed to ``strategy.on_blocked()``.

    Every platform gate uses one of these constants so strategy authors can
    programmatically identify why a signal was blocked and react accordingly.

    See ``BaseStrategy`` docstring for the full signal lifecycle and where
    each gate sits in the pipeline.
    """

    QUALITY_FILTER = "quality_filter"
    """Opportunity failed QualityFilterPipeline (ROI, liquidity, legs, etc.)."""

    RISK_DAILY_LOSS = "risk_daily_loss"
    """Daily realized loss exceeds the configured limit."""

    RISK_GROSS_EXPOSURE = "risk_gross_exposure"
    """Total open notional across all strategies exceeds the exposure cap."""

    RISK_CONSECUTIVE_LOSS = "risk_consecutive_loss"
    """Too many consecutive losing trades — circuit breaker tripped."""

    RISK_TRADE_NOTIONAL = "risk_trade_notional"
    """Single trade notional exceeds the per-trade cap."""

    RISK_OPEN_POSITIONS = "risk_open_positions"
    """Maximum number of open positions reached."""

    RISK_MARKET_EXPOSURE = "risk_market_exposure"
    """Per-market exposure cap reached (already have exposure on this market)."""

    TRADING_WINDOW = "trading_window"
    """Outside the configured UTC trading hours."""

    STACKING_GUARD = "stacking_guard"
    """Already have an open position on this market."""

    SIGNAL_EXPIRED = "signal_expired"
    """Signal is past its ``expires_at`` timestamp."""

    SCANNER_POOL_CAPACITY = "scanner_pool_capacity"
    """Scanner opportunity pool is at capacity."""

    DEDUPLICATION = "deduplication"
    """Duplicate signal collapsed with a higher-priority one."""
