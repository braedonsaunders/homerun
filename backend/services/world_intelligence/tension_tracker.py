"""GDELT-based country-pair tension tracking.

Uses the GDELT v2 API to compute rolling tension scores between
country pairs.  The Goldstein scale (-10 to +10) and article tone
provide the primary signal; event volume and escalation trends
(delta vs 7-day average) amplify the score.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx

from config import settings
from .tension_pair_catalog import tension_pair_catalog
from .taxonomy_catalog import taxonomy_catalog

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"

# GDELT publicly requests a minimum of one request every 5 seconds.
_GDELT_MIN_DELAY_SECONDS = 5.0
# Rate limiting: configurable minimum spacing between requests.
_RATE_LIMIT_DELAY_SECONDS = float(
    max(
        _GDELT_MIN_DELAY_SECONDS,
        getattr(settings, "WORLD_INTEL_GDELT_QUERY_DELAY_SECONDS", _GDELT_MIN_DELAY_SECONDS)
        or _GDELT_MIN_DELAY_SECONDS,
    )
)

# Rolling history retention
_HISTORY_MAX_DAYS = 90


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class CountryPairTension:
    """Tension score snapshot for a pair of countries."""

    country_a: str
    country_b: str
    tension_score: float  # 0-100
    event_count: int
    avg_goldstein_scale: float
    trend: str  # "rising" | "falling" | "stable"
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    top_event_types: list[str] = field(default_factory=list)


@dataclass
class _DailySnapshot:
    """Internal: one day's tension measurement for history tracking."""

    date: datetime
    tension_score: float
    event_count: int


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------


class TensionTracker:
    """Tracks geopolitical tension between country pairs via GDELT.

    The tension formula blends three components:
    1. **base_tone_score** - inverted average tone from GDELT articles
       (negative tone => higher tension).
    2. **event_volume_factor** - more articles about a pair means more
       attention / concern.
    3. **escalation_factor** - recent spike vs 7-day rolling average.
    """

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(timeout=30.0)

        # Current tension values keyed by (country_a, country_b) canonical pair
        self._current: dict[tuple[str, str], CountryPairTension] = {}

        # Rolling daily history per pair (up to 90 days)
        self._history: dict[tuple[str, str], list[_DailySnapshot]] = defaultdict(list)
        self._query_lock = asyncio.Lock()
        self._last_query_ts = 0.0
        self._last_error: Optional[str] = None

    def _default_pairs(self) -> list[tuple[str, str]]:
        configured = list(getattr(settings, "WORLD_INTEL_TENSION_PAIRS", []) or [])
        parsed: list[tuple[str, str]] = []
        for raw in configured:
            text = str(raw).strip().upper()
            if "-" in text:
                a, b = text.split("-", 1)
            elif ":" in text:
                a, b = text.split(":", 1)
            else:
                continue
            a = a.strip()
            b = b.strip()
            if len(a) == 2 and len(b) == 2 and a != b:
                parsed.append((a, b))
        if parsed:
            return parsed
        catalog_pairs = tension_pair_catalog.default_pairs()
        if catalog_pairs:
            return catalog_pairs
        return sorted(self._current.keys())

    @staticmethod
    def _gdelt_query_term(country: str) -> str:
        text = str(country or "").strip()
        if not text:
            return text
        lookup = tension_pair_catalog.query_names()
        return lookup.get(text.upper(), text)

    # -- Helpers -------------------------------------------------------------

    @staticmethod
    def _canonical_pair(a: str, b: str) -> tuple[str, str]:
        """Ensure consistent ordering of pair keys."""
        return (min(a, b), max(a, b))

    def _prune_history(self, pair: tuple[str, str]) -> None:
        """Remove snapshots older than 90 days."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=_HISTORY_MAX_DAYS)
        self._history[pair] = [
            s for s in self._history[pair] if s.date >= cutoff
        ]

    def _compute_trend(self, pair: tuple[str, str], current_score: float) -> str:
        """Determine if tension is rising, falling, or stable.

        Compares the current score to the 7-day average.  A delta
        greater than +5 is "rising", less than -5 is "falling",
        otherwise "stable".
        """
        snapshots = self._history.get(pair, [])
        if len(snapshots) < 3:
            return "stable"

        recent = snapshots[-7:]
        avg_7d = sum(s.tension_score for s in recent) / len(recent)
        delta = current_score - avg_7d
        if delta > 5.0:
            return "rising"
        elif delta < -5.0:
            return "falling"
        return "stable"

    def _compute_escalation_factor(self, pair: tuple[str, str], event_count: int) -> float:
        """Compute escalation multiplier based on volume spike.

        Returns a 0-25 bonus if today's event count exceeds the 7-day
        average by a large margin.
        """
        snapshots = self._history.get(pair, [])
        if len(snapshots) < 3:
            return 0.0

        recent = snapshots[-7:]
        avg_count = sum(s.event_count for s in recent) / len(recent)
        if avg_count <= 0:
            return 0.0

        ratio = event_count / avg_count
        # Ratio > 2x => escalation bonus (capped at 25 points)
        if ratio > 2.0:
            return min(25.0, (ratio - 1.0) * 10.0)
        return 0.0

    # -- GDELT fetching ------------------------------------------------------

    async def _fetch_pair_articles(
        self,
        country_a: str,
        country_b: str,
        timespan: str = "24h",
    ) -> list[dict]:
        """Query GDELT v2 doc API for articles mentioning both countries.

        Retries up to 2 times on HTTP 429 with exponential backoff.
        Returns a list of article dicts (may be empty).
        """
        query_a = self._gdelt_query_term(country_a).replace('"', "")
        query_b = self._gdelt_query_term(country_b).replace('"', "")
        query = f'"{query_a}" "{query_b}"'
        params: dict[str, str] = {
            "query": query,
            "mode": "artlist",
            "format": "json",
            "timespan": timespan,
            "sort": "datedesc",
            "maxrecords": "250",
        }

        max_retries = 2
        rate_limit_hint = "Please limit requests to one every 5 seconds"
        for attempt in range(max_retries + 1):
            try:
                await self._throttle_query()
                resp = await self._client.get(GDELT_DOC_API, params=params)
                if resp.status_code == 429:
                    if attempt < max_retries:
                        backoff = max(1.0, _RATE_LIMIT_DELAY_SECONDS) * (2 ** attempt)
                        logger.debug("GDELT 429 for %s-%s, retrying in %.0fs", country_a, country_b, backoff)
                        await asyncio.sleep(backoff)
                        continue
                    logger.warning("GDELT 429 for %s-%s after %d retries, skipping", country_a, country_b, max_retries)
                    self._last_error = f"429 rate-limited for {country_a}-{country_b}"
                    return []
                text = (resp.text or "").strip()
                if rate_limit_hint in text:
                    if attempt < max_retries:
                        backoff = max(_GDELT_MIN_DELAY_SECONDS, _RATE_LIMIT_DELAY_SECONDS) * (2 ** attempt)
                        logger.debug(
                            "GDELT soft rate-limit for %s-%s, retrying in %.0fs",
                            country_a,
                            country_b,
                            backoff,
                        )
                        await asyncio.sleep(backoff)
                        continue
                    logger.warning(
                        "GDELT soft rate-limit for %s-%s after %d retries, skipping",
                        country_a,
                        country_b,
                        max_retries,
                    )
                    self._last_error = f"soft rate-limited for {country_a}-{country_b}"
                    return []
                resp.raise_for_status()
                try:
                    data = resp.json()
                except ValueError as exc:
                    if attempt < max_retries:
                        backoff = max(_GDELT_MIN_DELAY_SECONDS, _RATE_LIMIT_DELAY_SECONDS) * (2 ** attempt)
                        logger.debug(
                            "GDELT non-JSON payload for %s-%s, retrying in %.0fs",
                            country_a,
                            country_b,
                            backoff,
                        )
                        await asyncio.sleep(backoff)
                        continue
                    logger.warning("GDELT invalid JSON for %s-%s: %s", country_a, country_b, exc)
                    self._last_error = f"invalid json for {country_a}-{country_b}"
                    return []
            except (httpx.HTTPError, ValueError) as exc:
                logger.warning("GDELT query failed for %s-%s: %s", country_a, country_b, exc)
                self._last_error = str(exc)
                return []

            articles = data.get("articles", [])
            self._last_error = None
            return articles if isinstance(articles, list) else []

        return []

    async def _throttle_query(self) -> None:
        if _RATE_LIMIT_DELAY_SECONDS <= 0:
            return
        async with self._query_lock:
            now = time.monotonic()
            wait = _RATE_LIMIT_DELAY_SECONDS - (now - self._last_query_ts)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_query_ts = time.monotonic()

    def _score_articles(self, articles: list[dict]) -> tuple[float, float, list[str]]:
        """Extract base tone score, event count, and top themes from articles.

        Returns:
            (base_tone_score 0-50, avg_goldstein, top_themes list)
        """
        if not articles:
            return 0.0, 0.0, []

        tones: list[float] = []
        themes: dict[str, int] = defaultdict(int)

        for art in articles:
            # GDELT tone field: "tone,positive,negative,polarity,..."
            tone_str = str(art.get("tone", "0"))
            try:
                tone_val = float(tone_str.split(",")[0])
                tones.append(tone_val)
            except (ValueError, IndexError):
                pass

            # Collect themes
            title = str(art.get("title", ""))
            for keyword in taxonomy_catalog.tension_title_keywords():
                if keyword in title.lower():
                    themes[keyword] += 1

        if not tones:
            return 0.0, 0.0, []

        avg_tone = sum(tones) / len(tones)
        # Invert and scale: very negative tone (-15) => high score (50)
        # Neutral (0) => 15, Positive (+5) => 5
        base_tone_score = max(0.0, min(50.0, 25.0 - avg_tone * 2.5))

        # Goldstein scale estimate from tone (rough proxy when not available)
        avg_goldstein = avg_tone * 0.7  # approximate correlation

        top_themes = sorted(themes.keys(), key=lambda k: themes[k], reverse=True)[:5]
        return base_tone_score, avg_goldstein, top_themes

    # -- Public API ----------------------------------------------------------

    async def update_tensions(
        self,
        country_pairs: Optional[list[tuple[str, str]]] = None,
    ) -> list[CountryPairTension]:
        """Refresh tension scores for the given country pairs.

        If *country_pairs* is ``None``, the default hot-pairs list is used.
        Queries GDELT for each pair, computes the composite tension score,
        stores the result, and returns all updated tensions.
        """
        pairs = country_pairs or self._default_pairs()
        concurrency = int(
            max(1, getattr(settings, "WORLD_INTEL_GDELT_MAX_CONCURRENCY", 3) or 3)
        )
        semaphore = asyncio.Semaphore(concurrency)

        async def _compute_pair(country_a: str, country_b: str) -> Optional[CountryPairTension]:
            async with semaphore:
                articles = await self._fetch_pair_articles(country_a, country_b)
                base_tone_score, avg_goldstein, top_themes = self._score_articles(articles)
                event_count = len(articles)

                if event_count > 0:
                    import math

                    volume_factor = min(35.0, 10.0 * math.log1p(event_count / 3.0))
                else:
                    volume_factor = 0.0

                pair = self._canonical_pair(country_a, country_b)
                escalation_factor = self._compute_escalation_factor(pair, event_count)
                tension_score = min(
                    100.0,
                    base_tone_score + volume_factor + escalation_factor,
                )
                trend = self._compute_trend(pair, tension_score)

                tension = CountryPairTension(
                    country_a=country_a,
                    country_b=country_b,
                    tension_score=round(tension_score, 1),
                    event_count=event_count,
                    avg_goldstein_scale=round(avg_goldstein, 2),
                    trend=trend,
                    last_updated=datetime.now(timezone.utc),
                    top_event_types=top_themes,
                )
                self._current[pair] = tension
                self._history[pair].append(
                    _DailySnapshot(
                        date=datetime.now(timezone.utc),
                        tension_score=tension_score,
                        event_count=event_count,
                    )
                )
                self._prune_history(pair)
                return tension

        computed = await asyncio.gather(
            *[_compute_pair(a, b) for a, b in pairs],
            return_exceptions=True,
        )
        results: list[CountryPairTension] = []
        for item in computed:
            if isinstance(item, CountryPairTension):
                results.append(item)
            elif isinstance(item, Exception):
                self._last_error = str(item)
                logger.debug("Tension pair computation error: %s", item)

        logger.info(
            "Tension tracker updated %d pairs (max=%.1f)",
            len(results),
            max((r.tension_score for r in results), default=0),
        )
        return results

    async def get_tension(self, country_a: str, country_b: str) -> Optional[CountryPairTension]:
        """Return the most recent tension score for a country pair.

        Returns ``None`` if the pair has not been tracked yet.
        """
        pair = self._canonical_pair(country_a, country_b)
        cached = self._current.get(pair)
        if cached is not None:
            return cached

        # Try a fresh fetch if we don't have it cached
        await self.update_tensions([(country_a, country_b)])
        return self._current.get(pair)

    def get_all_tensions(self) -> list[CountryPairTension]:
        """Return all currently tracked tension pairs."""
        return list(self._current.values())

    def get_high_tension_pairs(self, threshold: float = 60.0) -> list[CountryPairTension]:
        """Return all tracked pairs whose tension score is at or above *threshold*."""
        return [
            t for t in self._current.values()
            if t.tension_score >= threshold
        ]

    def get_tension_trajectory(
        self,
        country_a: str,
        country_b: str,
        days: int = 30,
    ) -> list[dict]:
        """Return recent daily tension values for sparkline rendering.

        Returns a list of ``{"date": ISO-str, "score": float, "events": int}``
        dicts, one per stored snapshot (up to *days* most recent).
        """
        pair = self._canonical_pair(country_a, country_b)
        snapshots = self._history.get(pair, [])
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        recent = [s for s in snapshots if s.date >= cutoff]
        return [
            {
                "date": s.date.isoformat(),
                "score": round(s.tension_score, 1),
                "events": s.event_count,
            }
            for s in recent
        ]

    def get_health(self) -> dict[str, object]:
        return {
            "enabled": True,
            "tracked_pairs": len(self._current),
            "request_spacing_seconds": _RATE_LIMIT_DELAY_SECONDS,
            "max_concurrency": int(
                max(1, getattr(settings, "WORLD_INTEL_GDELT_MAX_CONCURRENCY", 3) or 3)
            ),
            "last_error": self._last_error,
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

tension_tracker = TensionTracker()
