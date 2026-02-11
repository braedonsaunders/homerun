"""ACLED Armed Conflict Location & Event Data Project client.

Fetches structured conflict event data from the ACLED API
(https://api.acleddata.com/acled/read) for use in prediction market
signal generation. Covers battles, explosions, violence against civilians,
protests, riots, and strategic developments worldwide.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx

from config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACLED_API_URL = "https://api.acleddata.com/acled/read"

# Rate limiting: max 5 requests per minute
_RATE_LIMIT_MAX_REQUESTS = 5
_RATE_LIMIT_WINDOW_SECONDS = 60.0

# Circuit breaker
_CB_MAX_FAILURES = 5
_CB_COOLDOWN_SECONDS = 300.0  # 5 minutes

# Severity weights by event type (used in get_severity_score)
_EVENT_TYPE_WEIGHTS: dict[str, float] = {
    "violence against civilians": 1.0,
    "explosions/remote violence": 0.8,
    "battles": 0.7,
    "riots": 0.5,
    "protests": 0.3,
    "strategic developments": 0.2,
}

# Fields to request from the ACLED API
_ACLED_FIELDS = (
    "event_id_cnty|event_date|event_type|sub_event_type|country|iso3"
    "|latitude|longitude|fatalities|source|notes|timestamp"
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ConflictEvent:
    """A single ACLED conflict event."""

    event_id: str
    event_date: str
    event_type: str
    sub_event_type: str
    country: str
    iso3: str
    latitude: float
    longitude: float
    fatalities: int
    source: str
    notes: str
    timestamp: int  # ACLED unix timestamp

    @classmethod
    def from_api_row(cls, row: dict) -> ConflictEvent:
        """Parse a single row from the ACLED API response."""
        return cls(
            event_id=str(row.get("event_id_cnty", "")),
            event_date=str(row.get("event_date", "")),
            event_type=str(row.get("event_type", "")).lower(),
            sub_event_type=str(row.get("sub_event_type", "")),
            country=str(row.get("country", "")),
            iso3=str(row.get("iso3", "")),
            latitude=float(row.get("latitude", 0.0)),
            longitude=float(row.get("longitude", 0.0)),
            fatalities=int(row.get("fatalities", 0)),
            source=str(row.get("source", "")),
            notes=str(row.get("notes", "")),
            timestamp=int(row.get("timestamp", 0)),
        )


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class ACLEDClient:
    """Fetches and scores conflict events from the ACLED API.

    Features:
    - Circuit breaker pattern (backs off after consecutive failures)
    - Rate limiting with exponential backoff (max 5 req/min)
    - Optional API key authentication; falls back to public endpoint
    """

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(timeout=30.0)
        self._api_key: Optional[str] = getattr(settings, "ACLED_API_KEY", None)
        self._email: Optional[str] = getattr(settings, "ACLED_EMAIL", None)

        # Rate limiter state
        self._request_timestamps: list[float] = []

        # Circuit breaker state
        self._consecutive_failures: int = 0
        self._last_failure_at: float = 0.0

    # -- Rate limiting -------------------------------------------------------

    async def _wait_for_rate_limit(self) -> None:
        """Block until we are within the per-minute request budget."""
        now = time.monotonic()
        # Prune timestamps older than the rate-limit window
        self._request_timestamps = [
            ts for ts in self._request_timestamps
            if now - ts < _RATE_LIMIT_WINDOW_SECONDS
        ]
        if len(self._request_timestamps) >= _RATE_LIMIT_MAX_REQUESTS:
            oldest = self._request_timestamps[0]
            wait = _RATE_LIMIT_WINDOW_SECONDS - (now - oldest) + 0.1
            if wait > 0:
                logger.debug("ACLED rate limit: sleeping %.1fs", wait)
                await asyncio.sleep(wait)
        self._request_timestamps.append(time.monotonic())

    # -- Circuit breaker -----------------------------------------------------

    def _circuit_open(self) -> bool:
        """Return True if the circuit breaker is tripped."""
        if self._consecutive_failures < _CB_MAX_FAILURES:
            return False
        elapsed = time.monotonic() - self._last_failure_at
        if elapsed >= _CB_COOLDOWN_SECONDS:
            # Cooldown expired, allow a retry
            self._consecutive_failures = 0
            return False
        return True

    def _record_success(self) -> None:
        self._consecutive_failures = 0

    def _record_failure(self) -> None:
        self._consecutive_failures += 1
        self._last_failure_at = time.monotonic()

    # -- API methods ---------------------------------------------------------

    async def fetch_events(
        self,
        days_back: int = 7,
        country: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 500,
    ) -> list[ConflictEvent]:
        """Fetch conflict events from ACLED for the given time range.

        Args:
            days_back: Number of days into the past to query.
            country: Optional ISO-3166 country name filter.
            event_type: Optional ACLED event type filter.
            limit: Maximum rows to return (ACLED caps at 5000).

        Returns:
            List of parsed ConflictEvent instances, newest first.
        """
        if self._circuit_open():
            logger.warning("ACLED circuit breaker open, skipping request")
            return []

        await self._wait_for_rate_limit()

        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days_back)
        date_range = (
            f"{start_date.strftime('%Y-%m-%d')}"
            f"|{end_date.strftime('%Y-%m-%d')}"
        )

        params: dict[str, str | int] = {
            "event_date": date_range,
            "event_date_where": "BETWEEN",
            "limit": min(limit, 5000),
            "fields": _ACLED_FIELDS,
        }

        # Authenticated requests get higher rate limits
        if self._api_key and self._email:
            params["key"] = self._api_key
            params["email"] = self._email

        if country:
            params["country"] = country
        if event_type:
            params["event_type"] = event_type

        try:
            resp = await self._client.get(ACLED_API_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            self._record_failure()
            backoff = min(2 ** self._consecutive_failures, 60)
            logger.error(
                "ACLED API error (failure %d, backoff %ds): %s",
                self._consecutive_failures,
                backoff,
                exc,
            )
            return []

        self._record_success()

        rows = data.get("data", [])
        if not rows:
            logger.info("ACLED returned 0 events for range %s", date_range)
            return []

        events = []
        for row in rows:
            try:
                events.append(ConflictEvent.from_api_row(row))
            except (ValueError, TypeError) as exc:
                logger.debug("Skipping malformed ACLED row: %s", exc)
        logger.info("ACLED: fetched %d events (%d days back)", len(events), days_back)
        return events

    async def fetch_recent(self, hours: int = 24) -> list[ConflictEvent]:
        """Convenience wrapper for very recent events.

        ACLED's granularity is daily, so we fetch 1-2 days and filter
        by timestamp to approximate the requested hour window.
        """
        days = max(1, math.ceil(hours / 24))
        events = await self.fetch_events(days_back=days)
        if hours >= 24 * days:
            return events

        cutoff_ts = int((datetime.now(timezone.utc) - timedelta(hours=hours)).timestamp())
        return [e for e in events if e.timestamp >= cutoff_ts]

    # -- Aggregation helpers -------------------------------------------------

    @staticmethod
    def get_country_event_counts(
        events: list[ConflictEvent],
    ) -> dict[str, dict[str, int]]:
        """Aggregate events by country and event type.

        Returns:
            ``{"UA": {"battles": 12, "explosions/remote violence": 8}, ...}``
        """
        counts: dict[str, dict[str, int]] = {}
        for ev in events:
            by_type = counts.setdefault(ev.iso3, {})
            by_type[ev.event_type] = by_type.get(ev.event_type, 0) + 1
        return counts

    @staticmethod
    def get_severity_score(event: ConflictEvent) -> float:
        """Compute a 0-1 severity score for a single event.

        The score blends the event-type weight with a fatality-based
        multiplier.  A civilian-targeted event with many fatalities
        saturates at 1.0; a peaceful protest with no casualties scores
        near 0.3 * base.
        """
        type_weight = _EVENT_TYPE_WEIGHTS.get(event.event_type, 0.2)

        # Fatality multiplier: logarithmic scaling, 0 fatalities = 0.3 base,
        # 1 fatality = ~0.5, 10 = ~0.8, 50+ -> ~1.0
        if event.fatalities <= 0:
            fatality_weight = 0.3
        else:
            fatality_weight = min(1.0, 0.3 + 0.3 * math.log1p(event.fatalities))

        return min(1.0, type_weight * fatality_weight)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

acled_client = ACLEDClient()
