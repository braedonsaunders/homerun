"""World Intelligence RSS monitor.

Reads recent news articles from the NewsArticleCache DB table (populated by
the news_worker) and converts them into WorldSignals with signal_type="news".
No external HTTP fetching — relies entirely on the news_worker's persisted cache.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from .country_catalog import country_catalog

logger = logging.getLogger(__name__)

# Conflict/crisis keywords used to bump severity
_HIGH_SEVERITY_KEYWORDS = frozenset(
    {
        "war",
        "attack",
        "invasion",
        "conflict",
        "offensive",
        "airstrike",
        "bombing",
        "explosion",
        "troops",
        "missile",
        "nuclear",
        "ceasefire",
        "coup",
        "crisis",
        "emergency",
        "sanctions",
        "assassination",
        "hostages",
        "casualties",
        "killed",
        "deaths",
        "wounded",
    }
)

_MED_SEVERITY_KEYWORDS = frozenset(
    {
        "military",
        "tension",
        "protest",
        "unrest",
        "evacuation",
        "threat",
        "warning",
        "alert",
        "standoff",
        "blockade",
        "disputed",
        "escalation",
        "clashes",
        "riot",
        "strike",
    }
)

# Government-source agency names that get a severity boost
_GOV_AGENCIES = frozenset(
    {
        "white_house",
        "state_department",
        "defense",
        "treasury",
        "federal_reserve",
        "sec",
        "doj",
        "faa",
        "cdc",
    }
)


def _stable_news_signal_id(article_id: str) -> str:
    digest = hashlib.sha256(f"rss_news:{article_id}".encode()).hexdigest()[:20]
    return f"wi_{digest}"


def _infer_country(title: str, summary: str) -> Optional[str]:
    """Infer ISO3 country code from article text via country catalog lookup."""
    text = f"{title} {summary}".lower()
    # Walk all known country names and ISO3 codes
    alpha3_to_name = country_catalog.alpha3_to_name()
    name_to_alpha3 = country_catalog.name_to_alpha3()

    best: Optional[str] = None
    for name, iso3 in name_to_alpha3.items():
        if len(name) < 4:
            continue
        if name.lower() in text:
            best = iso3
            break
    if best:
        return best

    # Try ISO3 codes directly
    for iso3 in alpha3_to_name:
        if f" {iso3.lower()} " in f" {text} ":
            return iso3

    return None


def _compute_severity(title: str, summary: str, feed_source: str, category: str) -> float:
    """Compute 0-1 severity score for a news article."""
    text = f"{title} {summary}".lower()

    severity = 0.35  # baseline for non-gov

    # Government source boost
    if feed_source in ("gov_rss", "government"):
        severity = 0.45
        if category in ("critical",):
            severity = 0.65
        elif category in ("high",):
            severity = 0.55

    # Keyword boosts (additive, capped)
    for kw in _HIGH_SEVERITY_KEYWORDS:
        if kw in text:
            severity += 0.12
            break
    for kw in _MED_SEVERITY_KEYWORDS:
        if kw in text:
            severity += 0.06
            break

    return min(0.85, severity)


class WorldIntelRSSMonitor:
    """Reads NewsArticleCache DB entries and emits WorldSignal objects.

    No external HTTP calls are made. The news_worker populates the DB table;
    we just read recent entries as an additional signal source for world intel.
    """

    def __init__(self) -> None:
        self._last_count = 0
        self._last_error: Optional[str] = None
        self._last_ok = True

    async def fetch_signals(
        self,
        max_age_hours: float = 3.0,
        limit: int = 200,
    ) -> list[Any]:
        """Return WorldSignal-like dicts from recent news articles.

        Returns an empty list (not an exception) if the news table is empty
        or unavailable — world intel degrades gracefully without RSS.
        """

        # Import here to avoid circular imports at module level
        from models.database import AsyncSessionLocal, NewsArticleCache
        from sqlalchemy import select

        # We return plain dicts that match the WorldSignal interface so
        # signal_aggregator can call _news_to_signal() on them.
        articles: list[dict] = []
        cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        cutoff_naive = cutoff.replace(tzinfo=None)

        try:
            async with AsyncSessionLocal() as session:
                stmt = (
                    select(NewsArticleCache)
                    .where(NewsArticleCache.fetched_at >= cutoff_naive)
                    .order_by(NewsArticleCache.fetched_at.desc())
                    .limit(limit)
                )
                result = await session.execute(stmt)
                rows = result.scalars().all()

            for row in rows:
                title = str(row.title or "").strip()
                if not title:
                    continue
                summary = str(row.summary or "").strip()
                source = str(row.source or row.feed_source or "rss").strip()
                feed_source = str(row.feed_source or "").strip()
                category = str(row.category or "").strip()

                country = _infer_country(title, summary)
                severity = _compute_severity(title, summary, feed_source, category)

                fetched_at = row.fetched_at
                if isinstance(fetched_at, datetime) and fetched_at.tzinfo is None:
                    fetched_at = fetched_at.replace(tzinfo=timezone.utc)

                published = row.published
                if isinstance(published, datetime) and published.tzinfo is None:
                    published = published.replace(tzinfo=timezone.utc)
                detected_at = published or fetched_at or datetime.now(timezone.utc)

                articles.append(
                    {
                        "article_id": row.article_id,
                        "title": title,
                        "summary": summary,
                        "source": source,
                        "feed_source": feed_source,
                        "category": category,
                        "country": country,
                        "severity": severity,
                        "detected_at": detected_at,
                        "url": str(row.url or "").strip(),
                    }
                )

            self._last_count = len(articles)
            self._last_error = None
            self._last_ok = True

        except Exception as exc:
            logger.warning("WorldIntelRSSMonitor failed to read NewsArticleCache: %s", exc)
            self._last_error = str(exc)
            self._last_ok = False

        return articles

    def get_health(self) -> dict[str, Any]:
        return {
            "ok": self._last_ok,
            "count": self._last_count,
            "last_error": self._last_error,
        }


rss_monitor = WorldIntelRSSMonitor()
