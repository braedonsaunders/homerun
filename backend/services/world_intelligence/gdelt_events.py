"""GDELT Event Database client for structured event extraction.

Complements the existing GDELT DOC 2.0 article API with structured
event data using the CAMEO event ontology. Provides country-pair
tension scoring and event taxonomy classification.
"""

from __future__ import annotations

import asyncio
import csv
import io
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx

from config import settings
from .tension_pair_catalog import tension_pair_catalog

logger = logging.getLogger(__name__)

_HTTP_TIMEOUT = 20
_USER_AGENT = "Mozilla/5.0 (compatible; Homerun/2.0)"

# GDELT GKG (Global Knowledge Graph) API
GDELT_GKG_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

# CAMEO event code categories relevant to prediction markets
CAMEO_CATEGORIES = {
    # Verbal cooperation
    "01": ("make_statement", 0.0),
    "02": ("appeal", 0.1),
    "03": ("express_intent_cooperate", -0.2),
    "04": ("consult", -0.1),
    "05": ("diplomatic_cooperation", -0.3),
    # Verbal conflict
    "06": ("reject", 0.2),
    "07": ("threaten", 0.5),
    "08": ("protest", 0.4),
    "09": ("investigate", 0.1),
    # Material cooperation
    "10": ("demand", 0.3),
    "11": ("disapprove", 0.3),
    "12": ("reduce_relations", 0.5),
    "13": ("threaten_force", 0.7),
    "14": ("protest_violently", 0.6),
    # Material conflict
    "15": ("exhibit_force", 0.7),
    "16": ("reduce_relations_force", 0.8),
    "17": ("coerce", 0.8),
    "18": ("assault", 0.9),
    "19": ("fight", 0.95),
    "20": ("mass_violence", 1.0),
}

# Goldstein scale: -10 (most conflictual) to +10 (most cooperative)
# We normalize to 0-100 tension score where 100 = maximum tension


@dataclass
class GDELTEvent:
    """A structured event from GDELT."""

    event_id: str
    event_date: datetime
    actor1_country: str
    actor2_country: str
    cameo_code: str
    cameo_category: str
    goldstein_scale: float  # -10 to +10
    num_mentions: int
    num_sources: int
    avg_tone: float
    source_url: str
    tension_contribution: float  # 0-1 computed


@dataclass
class CountryPairEventSummary:
    """Summary of events between two countries."""

    country_a: str
    country_b: str
    total_events: int
    avg_goldstein: float
    avg_tone: float
    tension_score: float  # 0-100
    top_cameo_categories: list[str]
    event_trend: str  # escalating, de-escalating, stable
    period_days: int


class GDELTEventService:
    """Extracts structured events and tension scores from GDELT."""

    def __init__(self) -> None:
        self._event_cache: dict[str, GDELTEvent] = {}
        self._pair_summaries: dict[str, CountryPairEventSummary] = {}
        self._last_fetch_at: Optional[datetime] = None

    @staticmethod
    def _query_term(country: str) -> str:
        lookup = tension_pair_catalog.query_names()
        text = str(country or "").strip()
        if not text:
            return text
        return lookup.get(text.upper(), text)

    async def fetch_country_pair_events(
        self,
        country_a: str,
        country_b: str,
        days_back: int = 7,
    ) -> list[GDELTEvent]:
        """Fetch events between two countries from GDELT."""
        query_a = self._query_term(country_a).replace('"', "")
        query_b = self._query_term(country_b).replace('"', "")
        query = f'"{query_a}" "{query_b}"'
        timespan = f"{days_back * 24}h"

        try:
            import urllib.parse

            encoded = urllib.parse.quote(query)
            url = (
                f"{GDELT_GKG_URL}?query={encoded}"
                f"&mode=artlist&maxrecords=100&format=json"
                f"&sort=datedesc&timespan={timespan}"
            )

            async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
                resp = await client.get(url, headers={"User-Agent": _USER_AGENT})
                if resp.status_code != 200:
                    return []

            data = resp.json()
            articles = data.get("articles", [])
            events: list[GDELTEvent] = []

            for art in articles:
                tone = float(art.get("tone", "0").split(",")[0]) if art.get("tone") else 0.0
                event = GDELTEvent(
                    event_id=art.get("url", "")[:64],
                    event_date=_parse_gdelt_date(art.get("seendate", "")),
                    actor1_country=country_a,
                    actor2_country=country_b,
                    cameo_code="",  # Not directly available from DOC API
                    cameo_category=_classify_tone(tone),
                    goldstein_scale=_tone_to_goldstein(tone),
                    num_mentions=1,
                    num_sources=1,
                    avg_tone=tone,
                    source_url=art.get("url", ""),
                    tension_contribution=_tone_to_tension(tone),
                )
                events.append(event)

            return events

        except Exception as e:
            logger.debug("GDELT event fetch failed for %s-%s: %s", country_a, country_b, e)
            return []

    async def compute_pair_tension(
        self,
        country_a: str,
        country_b: str,
        days_back: int = 7,
    ) -> CountryPairEventSummary:
        """Compute tension score between two countries."""
        events = await self.fetch_country_pair_events(country_a, country_b, days_back)

        if not events:
            return CountryPairEventSummary(
                country_a=country_a,
                country_b=country_b,
                total_events=0,
                avg_goldstein=0.0,
                avg_tone=0.0,
                tension_score=0.0,
                top_cameo_categories=[],
                event_trend="stable",
                period_days=days_back,
            )

        avg_goldstein = sum(e.goldstein_scale for e in events) / len(events)
        avg_tone = sum(e.avg_tone for e in events) / len(events)

        # Tension score: normalize Goldstein from [-10,+10] to [0,100]
        # -10 (most conflictual) -> 100 tension
        # +10 (most cooperative) -> 0 tension
        base_tension = max(0, min(100, ((-avg_goldstein + 10) / 20) * 100))

        # Volume factor: more events = higher confidence in tension score
        volume_factor = min(1.5, 1.0 + len(events) / 100)

        # Tone factor: negative tone amplifies tension
        tone_factor = 1.0
        if avg_tone < -5:
            tone_factor = 1.3
        elif avg_tone < -2:
            tone_factor = 1.15
        elif avg_tone > 2:
            tone_factor = 0.85

        tension_score = min(100, base_tension * volume_factor * tone_factor)

        # Categorize events
        categories: dict[str, int] = {}
        for e in events:
            cat = e.cameo_category
            categories[cat] = categories.get(cat, 0) + 1
        top_cats = sorted(categories, key=categories.get, reverse=True)[:5]

        # Determine trend by comparing first half vs second half
        mid = len(events) // 2
        if mid > 0:
            first_half_tone = sum(e.avg_tone for e in events[mid:]) / max(1, len(events) - mid)
            second_half_tone = sum(e.avg_tone for e in events[:mid]) / max(1, mid)
            if second_half_tone < first_half_tone - 1:
                trend = "escalating"
            elif second_half_tone > first_half_tone + 1:
                trend = "de-escalating"
            else:
                trend = "stable"
        else:
            trend = "stable"

        summary = CountryPairEventSummary(
            country_a=country_a,
            country_b=country_b,
            total_events=len(events),
            avg_goldstein=round(avg_goldstein, 2),
            avg_tone=round(avg_tone, 2),
            tension_score=round(tension_score, 1),
            top_cameo_categories=top_cats,
            event_trend=trend,
            period_days=days_back,
        )

        pair_key = f"{country_a}_{country_b}"
        self._pair_summaries[pair_key] = summary
        return summary

    async def compute_all_default_pairs(self) -> list[CountryPairEventSummary]:
        """Compute tensions for all default country pairs."""
        pairs = tension_pair_catalog.default_pairs()
        if not pairs:
            return []

        tasks = [self.compute_pair_tension(a, b) for a, b in pairs]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        summaries = []
        for result in results:
            if isinstance(result, CountryPairEventSummary):
                summaries.append(result)
        return summaries

    def get_all_summaries(self) -> list[CountryPairEventSummary]:
        """Get all cached pair summaries."""
        return sorted(
            self._pair_summaries.values(),
            key=lambda s: s.tension_score,
            reverse=True,
        )


def _parse_gdelt_date(date_str: str) -> datetime:
    if not date_str:
        return datetime.now(timezone.utc)
    try:
        cleaned = date_str.replace("Z", "").replace("z", "")
        return datetime.strptime(cleaned, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return datetime.now(timezone.utc)


def _classify_tone(tone: float) -> str:
    """Classify GDELT tone value into a category."""
    if tone < -8:
        return "highly_conflictual"
    elif tone < -4:
        return "conflictual"
    elif tone < -1:
        return "mildly_negative"
    elif tone < 1:
        return "neutral"
    elif tone < 4:
        return "mildly_positive"
    elif tone < 8:
        return "cooperative"
    else:
        return "highly_cooperative"


def _tone_to_goldstein(tone: float) -> float:
    """Approximate Goldstein scale from GDELT tone."""
    # GDELT tone ranges roughly -25 to +25, Goldstein -10 to +10
    return max(-10.0, min(10.0, tone * 0.4))


def _tone_to_tension(tone: float) -> float:
    """Convert tone to tension contribution (0-1)."""
    # Negative tone = higher tension
    return max(0.0, min(1.0, (-tone + 10) / 20))


# Singleton
gdelt_event_service = GDELTEventService()
