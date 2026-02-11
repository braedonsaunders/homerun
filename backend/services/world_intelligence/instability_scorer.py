"""Country Instability Index (CII).

Composite 0-100 instability score per country, tuned for prediction
market relevance.  Blends structural baseline risk with real-time event
data from ACLED, military monitors, and news velocity to produce an
actionable score that correlates with geopolitical market price moves.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Structural constants
# ---------------------------------------------------------------------------

# Regime multipliers: authoritarian regimes amplify event impact.
# Sources: V-Dem, Freedom House.  Values are rough tiers.
REGIME_MULTIPLIERS: dict[str, float] = {
    # Authoritarian (2.0-2.5)
    "PRK": 2.5, "ERI": 2.3, "TKM": 2.3, "SYR": 2.2, "AFG": 2.2,
    "MMR": 2.0, "CHN": 2.0, "RUS": 2.0, "IRN": 2.0, "BLR": 2.0,
    "SAU": 2.0, "CUB": 2.0, "VEN": 2.0,
    # Hybrid (1.2-1.5)
    "TUR": 1.5, "PAK": 1.4, "NGA": 1.3, "BGD": 1.3, "PHL": 1.2,
    "HUN": 1.2, "MEX": 1.2, "IDN": 1.2, "THA": 1.3, "UKR": 1.2,
    # Democracies (0.3-0.5) — events in democracies are less destabilising
    "USA": 0.4, "GBR": 0.3, "DEU": 0.3, "FRA": 0.4, "JPN": 0.3,
    "CAN": 0.3, "AUS": 0.3, "KOR": 0.4, "IND": 0.5, "BRA": 0.5,
    "ZAF": 0.5, "ISR": 0.5,
}
_DEFAULT_REGIME_MULTIPLIER = 1.0

# Structural baseline risk (0-40 range).
BASELINE_RISK: dict[str, float] = {
    # Failed / near-failed states
    "SOM": 40, "YEM": 38, "SSD": 38, "AFG": 37, "SYR": 36, "LBY": 35,
    "COD": 35, "HTI": 34, "MMR": 33, "SDN": 35,
    # Active conflict zones
    "UKR": 30, "IRQ": 28, "MLI": 27, "BFA": 27, "NGA": 26, "ETH": 26,
    "MOZ": 25, "PSE": 30, "LBN": 25,
    # Developing / moderate risk
    "PAK": 22, "EGY": 20, "VEN": 20, "COL": 18, "PHL": 18,
    "THA": 16, "MEX": 17, "IRN": 22, "PRK": 25, "RUS": 20,
    "TUR": 18, "BGD": 18, "CHN": 15, "SAU": 16,
    # Stable
    "USA": 8, "GBR": 5, "DEU": 5, "FRA": 7, "JPN": 5,
    "CAN": 5, "AUS": 5, "KOR": 8, "IND": 15, "BRA": 14,
    "ISR": 18, "ZAF": 14, "IDN": 12,
}
_DEFAULT_BASELINE = 12.0

# UCDP conflict floors: countries with active armed conflict get a score floor
UCDP_ACTIVE_WAR_FLOOR = 70.0
UCDP_MINOR_CONFLICT_FLOOR = 50.0

UCDP_ACTIVE_WARS: set[str] = {"UKR", "SDN", "MMR", "PSE", "SYR", "YEM"}
UCDP_MINOR_CONFLICTS: set[str] = {
    "ETH", "NGA", "MLI", "BFA", "MOZ", "COD", "SOM", "IRQ", "AFG",
}

# Rolling history retention
_HISTORY_MAX_DAYS = 30


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class CountryInstabilityScore:
    """Instability score for a single country."""

    country: str
    iso3: str
    score: float  # 0-100
    components: dict[str, float]  # sub-score breakdown
    trend: str  # "rising" | "falling" | "stable"
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    change_24h: float = 0.0
    change_7d: float = 0.0
    contributing_signals: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class _HistoricalScore:
    """Internal: daily score snapshot for trend calculation."""
    date: datetime
    score: float


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


class InstabilityScorer:
    """Computes and tracks Country Instability Index scores.

    Score formula:
        final = baseline_risk * 0.4 + event_score * 0.6

    Where event_score is a regime-multiplied blend of:
        - unrest_component    (25%) - protests/riots
        - conflict_component  (30%) - ACLED violent events
        - security_component  (20%) - military activity
        - information_component (25%) - news velocity
    """

    def __init__(self) -> None:
        # Latest scores per country ISO3
        self._scores: dict[str, CountryInstabilityScore] = {}
        # Rolling daily history for trend calculation
        self._history: dict[str, list[_HistoricalScore]] = defaultdict(list)

    # -- Component scorers ---------------------------------------------------

    @staticmethod
    def _unrest_component(protest_count: int, riot_count: int, multiplier: float) -> float:
        """Compute unrest sub-score (0-100) from protest + riot counts.

        Uses logarithmic scaling: a handful of protests is common,
        but dozens in one day is highly unusual.
        """
        raw = (protest_count + riot_count * 2) * multiplier
        if raw <= 0:
            return 0.0
        return min(100.0, 30.0 * math.log1p(raw))

    @staticmethod
    def _conflict_component(
        civilian_violence_count: int,
        explosions_count: int,
        battles_count: int,
    ) -> float:
        """Compute conflict sub-score (0-100) from ACLED violent events.

        Weighted: civilian violence (5x), explosions (4x), battles (3x).
        """
        weighted = (
            civilian_violence_count * 5
            + explosions_count * 4
            + battles_count * 3
        )
        if weighted <= 0:
            return 0.0
        return min(100.0, 25.0 * math.log1p(weighted))

    @staticmethod
    def _security_component(flight_count: int, vessel_count: int) -> float:
        """Compute security sub-score (0-100) from military activity.

        Flights contribute 3 points each; vessels contribute 5 each.
        """
        raw = flight_count * 3 + vessel_count * 5
        return min(100.0, float(raw))

    @staticmethod
    def _information_component(articles_per_hour: float) -> float:
        """Compute information sub-score (0-100) from news velocity.

        Uses logarithmic scaling: 1 article/hr = ~0, 10 = ~30,
        100 = ~60, 1000+ = ~90.
        """
        if articles_per_hour <= 0:
            return 0.0
        return min(100.0, 20.0 * math.log1p(articles_per_hour))

    # -- Trend helpers -------------------------------------------------------

    def _compute_trend(self, iso3: str, current_score: float) -> str:
        """Determine trend by comparing to 7-day average."""
        history = self._history.get(iso3, [])
        if len(history) < 3:
            return "stable"
        recent = history[-7:]
        avg_7d = sum(h.score for h in recent) / len(recent)
        delta = current_score - avg_7d
        if delta > 5.0:
            return "rising"
        elif delta < -5.0:
            return "falling"
        return "stable"

    def _compute_changes(self, iso3: str, current_score: float) -> tuple[float, float]:
        """Return (change_24h, change_7d) deltas."""
        history = self._history.get(iso3, [])
        change_24h = 0.0
        change_7d = 0.0

        if history:
            change_24h = current_score - history[-1].score

        if len(history) >= 7:
            change_7d = current_score - history[-7].score

        return round(change_24h, 1), round(change_7d, 1)

    def _prune_history(self, iso3: str) -> None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=_HISTORY_MAX_DAYS)
        self._history[iso3] = [
            h for h in self._history[iso3] if h.date >= cutoff
        ]

    # -- Public API ----------------------------------------------------------

    async def compute_scores(
        self,
        conflict_events: Optional[list] = None,
        military_events: Optional[list] = None,
        news_velocity: Optional[dict[str, float]] = None,
        protest_events: Optional[list] = None,
    ) -> dict[str, CountryInstabilityScore]:
        """Compute instability scores for all countries with available data.

        Args:
            conflict_events: List of ConflictEvent from ACLED.
            military_events: List of MilitaryActivity from military_monitor.
            news_velocity: Dict of ISO3 -> articles per hour.
            protest_events: Separate protest/riot events if available.

        Returns:
            Dict of ISO3 -> CountryInstabilityScore.
        """
        conflict_events = conflict_events or []
        military_events = military_events or []
        news_velocity = news_velocity or {}
        protest_events = protest_events or []

        # Aggregate conflict events by country and type
        country_conflicts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for ev in conflict_events:
            iso3 = getattr(ev, "iso3", "")
            etype = getattr(ev, "event_type", "")
            if iso3:
                country_conflicts[iso3][etype] += 1

        # Aggregate protests (may overlap with conflict_events)
        country_protests: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for ev in protest_events:
            iso3 = getattr(ev, "iso3", "")
            etype = getattr(ev, "event_type", "")
            if iso3:
                country_protests[iso3][etype] += 1

        # Aggregate military events by country
        country_military: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for ev in military_events:
            country = getattr(ev, "country", "")
            atype = getattr(ev, "activity_type", "")
            if country:
                country_military[country][atype] += 1

        # Collect all countries that appear in any data source
        all_countries: set[str] = set()
        all_countries.update(country_conflicts.keys())
        all_countries.update(country_protests.keys())
        all_countries.update(country_military.keys())
        all_countries.update(news_velocity.keys())

        results: dict[str, CountryInstabilityScore] = {}

        for iso3 in all_countries:
            baseline = BASELINE_RISK.get(iso3, _DEFAULT_BASELINE)
            multiplier = REGIME_MULTIPLIERS.get(iso3, _DEFAULT_REGIME_MULTIPLIER)

            # Unrest component (25%)
            protests = country_protests.get(iso3, {})
            # Also check conflict_events for protest/riot types
            conflicts = country_conflicts.get(iso3, {})
            protest_count = protests.get("protests", 0) + conflicts.get("protests", 0)
            riot_count = protests.get("riots", 0) + conflicts.get("riots", 0)
            unrest = self._unrest_component(protest_count, riot_count, multiplier)

            # Conflict component (30%)
            civilian_v = conflicts.get("violence against civilians", 0)
            explosions = conflicts.get("explosions/remote violence", 0)
            battles = conflicts.get("battles", 0)
            conflict_score = self._conflict_component(civilian_v, explosions, battles)

            # Security component (20%)
            mil = country_military.get(iso3, {})
            flights = mil.get("flight", 0)
            vessels = mil.get("vessel", 0)
            security = self._security_component(flights, vessels)

            # Information component (25%)
            articles_per_hour = news_velocity.get(iso3, 0.0)
            information = self._information_component(articles_per_hour)

            # Weighted event score
            event_score = (
                unrest * 0.25
                + conflict_score * 0.30
                + security * 0.20
                + information * 0.25
            )

            # Apply regime multiplier to event_score
            event_score = min(100.0, event_score * multiplier)

            # Final blended score
            score = baseline * 0.4 + event_score * 0.6

            # Apply UCDP conflict floors
            if iso3 in UCDP_ACTIVE_WARS:
                score = max(score, UCDP_ACTIVE_WAR_FLOOR)
            elif iso3 in UCDP_MINOR_CONFLICTS:
                score = max(score, UCDP_MINOR_CONFLICT_FLOOR)

            score = min(100.0, round(score, 1))

            contributing: list[dict[str, Any]] = []
            if protest_count + riot_count > 0:
                contributing.append({
                    "type": "unrest",
                    "detail": f"{protest_count} protests, {riot_count} riots",
                    "sub_score": round(unrest, 1),
                })
            if civilian_v + explosions + battles > 0:
                contributing.append({
                    "type": "conflict",
                    "detail": f"{civilian_v} civ, {explosions} exp, {battles} bat",
                    "sub_score": round(conflict_score, 1),
                })
            if flights + vessels > 0:
                contributing.append({
                    "type": "security",
                    "detail": f"{flights} flights, {vessels} vessels",
                    "sub_score": round(security, 1),
                })
            if articles_per_hour > 0:
                contributing.append({
                    "type": "information",
                    "detail": f"{articles_per_hour:.1f} articles/hr",
                    "sub_score": round(information, 1),
                })

            trend = self._compute_trend(iso3, score)
            change_24h, change_7d = self._compute_changes(iso3, score)

            result = CountryInstabilityScore(
                country=iso3,  # human-readable name not always available
                iso3=iso3,
                score=score,
                components={
                    "baseline": round(baseline, 1),
                    "unrest": round(unrest, 1),
                    "conflict": round(conflict_score, 1),
                    "security": round(security, 1),
                    "information": round(information, 1),
                    "event_score_raw": round(event_score, 1),
                },
                trend=trend,
                change_24h=change_24h,
                change_7d=change_7d,
                contributing_signals=contributing,
            )

            self._scores[iso3] = result
            self._history[iso3].append(
                _HistoricalScore(date=datetime.now(timezone.utc), score=score)
            )
            self._prune_history(iso3)
            results[iso3] = result

        logger.info(
            "Instability scores computed for %d countries (max=%.1f)",
            len(results),
            max((r.score for r in results.values()), default=0),
        )
        return results

    async def get_score(self, country_iso3: str) -> Optional[CountryInstabilityScore]:
        """Return the latest instability score for a country, if available."""
        return self._scores.get(country_iso3)

    def get_all_scores(self) -> dict[str, CountryInstabilityScore]:
        """Return all current instability scores keyed by ISO3."""
        return dict(self._scores)

    def get_critical_countries(self, threshold: float = 60.0) -> list[CountryInstabilityScore]:
        """Return all countries whose instability score meets or exceeds *threshold*."""
        return sorted(
            [s for s in self._scores.values() if s.score >= threshold],
            key=lambda s: s.score,
            reverse=True,
        )

    def get_score_changes(self, timeframe_hours: int = 24) -> list[dict]:
        """Return countries with the biggest score movements in the given timeframe.

        Sorted by absolute change descending.
        """
        changes: list[dict] = []
        for iso3, score_obj in self._scores.items():
            delta = score_obj.change_24h if timeframe_hours <= 24 else score_obj.change_7d
            if abs(delta) > 1.0:
                changes.append({
                    "country": iso3,
                    "score": score_obj.score,
                    "change": delta,
                    "trend": score_obj.trend,
                })
        return sorted(changes, key=lambda c: abs(c["change"]), reverse=True)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

instability_scorer = InstabilityScorer()
