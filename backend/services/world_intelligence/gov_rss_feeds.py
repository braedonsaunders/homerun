"""US Government RSS feed aggregation.

Monitors official government sources for policy announcements,
regulatory actions, and executive communications that directly
affect prediction markets. These feeds provide 5-15 minute latency
advantage over general news aggregators.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from xml.etree import ElementTree

import httpx

from config import settings

logger = logging.getLogger(__name__)

_HTTP_TIMEOUT = 15
_USER_AGENT = "Mozilla/5.0 (compatible; Homerun/2.0)"

# Official US Government RSS feeds organized by market relevance
GOV_RSS_FEEDS: dict[str, list[dict[str, str]]] = {
    "white_house": [
        {"url": "https://www.whitehouse.gov/feed/", "name": "White House Blog", "priority": "high"},
        {"url": "https://www.whitehouse.gov/briefing-room/statements-releases/feed/", "name": "WH Statements", "priority": "critical"},
        {"url": "https://www.whitehouse.gov/briefing-room/presidential-actions/feed/", "name": "Presidential Actions", "priority": "critical"},
    ],
    "state_department": [
        {"url": "https://www.state.gov/rss-feed/press-releases/feed/", "name": "State Dept Press", "priority": "high"},
        {"url": "https://www.state.gov/rss-feed/travel-advisories/feed/", "name": "Travel Advisories", "priority": "medium"},
    ],
    "defense": [
        {"url": "https://www.defense.gov/DesktopModules/ArticleCS/RSS.ashx?ContentType=1&Site=945", "name": "DoD News", "priority": "high"},
    ],
    "treasury": [
        {"url": "https://home.treasury.gov/system/files/136/treasury-rss.xml", "name": "Treasury", "priority": "high"},
    ],
    "federal_reserve": [
        {"url": "https://www.federalreserve.gov/feeds/press_all.xml", "name": "Fed Press Releases", "priority": "critical"},
        {"url": "https://www.federalreserve.gov/feeds/press_monetary.xml", "name": "Fed Monetary Policy", "priority": "critical"},
    ],
    "sec": [
        {"url": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=&dateb=&owner=include&count=20&search_text=&start=0&output=atom", "name": "SEC EDGAR Filings", "priority": "medium"},
        {"url": "https://www.sec.gov/news/pressreleases.rss", "name": "SEC Press Releases", "priority": "high"},
    ],
    "justice": [
        {"url": "https://www.justice.gov/feeds/opa/justice-news.xml", "name": "DOJ News", "priority": "high"},
    ],
    "cdc": [
        {"url": "https://tools.cdc.gov/api/v2/resources/media/rss?topic=outbreaks", "name": "CDC Outbreaks", "priority": "high"},
    ],
    "faa": [
        {"url": "https://www.faa.gov/rss/all", "name": "FAA Notices", "priority": "medium"},
    ],
}


@dataclass
class GovArticle:
    """Article from a government RSS feed."""

    article_id: str
    title: str
    url: str
    source: str
    agency: str  # white_house, state_department, etc.
    priority: str  # critical, high, medium
    published: Optional[datetime] = None
    summary: str = ""
    feed_source: str = "gov_rss"
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class GovRSSFeedService:
    """Fetches and aggregates US government RSS feeds."""

    def __init__(self) -> None:
        self._articles: dict[str, GovArticle] = {}
        self._last_fetch_at: Optional[datetime] = None

    async def fetch_all(self) -> list[GovArticle]:
        """Fetch from all enabled government RSS feeds. Returns NEW articles only."""
        if not settings.WORLD_INTEL_GOV_RSS_ENABLED:
            return []

        new_articles: list[GovArticle] = []
        tasks = []

        for agency, feeds in GOV_RSS_FEEDS.items():
            for feed_info in feeds:
                tasks.append(self._fetch_single_feed(agency, feed_info))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                logger.debug("Gov RSS fetch error: %s", result)
                continue
            if isinstance(result, list):
                for article in result:
                    if article.article_id not in self._articles:
                        self._articles[article.article_id] = article
                        new_articles.append(article)

        self._last_fetch_at = datetime.now(timezone.utc)
        self._prune_old_articles()

        if new_articles:
            logger.info(
                "Gov RSS: %d new articles from %d agencies",
                len(new_articles),
                len(set(a.agency for a in new_articles)),
            )
        return new_articles

    async def _fetch_single_feed(
        self, agency: str, feed_info: dict
    ) -> list[GovArticle]:
        """Fetch articles from a single government RSS feed."""
        url = feed_info["url"]
        name = feed_info["name"]
        priority = feed_info.get("priority", "medium")

        try:
            async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
                resp = await client.get(
                    url,
                    headers={"User-Agent": _USER_AGENT},
                    follow_redirects=True,
                )
                if resp.status_code != 200:
                    return []

                root = ElementTree.fromstring(resp.text)
                articles: list[GovArticle] = []

                # Try RSS format
                items = root.findall(".//item")
                if not items:
                    # Try Atom format
                    ns = {"atom": "http://www.w3.org/2005/Atom"}
                    items = root.findall(".//atom:entry", ns)

                for item in items[:20]:
                    title = (
                        item.findtext("title", "").strip()
                        or item.findtext("{http://www.w3.org/2005/Atom}title", "").strip()
                    )
                    link = item.findtext("link", "").strip()
                    if not link:
                        atom_link = item.find("{http://www.w3.org/2005/Atom}link")
                        if atom_link is not None:
                            link = atom_link.get("href", "")
                    pub_date = (
                        item.findtext("pubDate", "").strip()
                        or item.findtext("{http://www.w3.org/2005/Atom}published", "").strip()
                        or item.findtext("{http://www.w3.org/2005/Atom}updated", "").strip()
                    )
                    description = (
                        item.findtext("description", "").strip()
                        or item.findtext("{http://www.w3.org/2005/Atom}summary", "").strip()
                    )

                    if not link or not title:
                        continue

                    article_id = hashlib.sha256(link.encode()).hexdigest()[:16]
                    published = _parse_date(pub_date)
                    summary = _strip_html(description)[:500] if description else ""

                    articles.append(
                        GovArticle(
                            article_id=article_id,
                            title=title,
                            url=link,
                            source=name,
                            agency=agency,
                            priority=priority,
                            published=published,
                            summary=summary,
                        )
                    )

                return articles

        except Exception as e:
            logger.debug("Gov RSS fetch failed for '%s' (%s): %s", name, url, e)
            return []

    def get_articles(self, max_age_hours: int = 48) -> list[GovArticle]:
        """Get all articles, optionally filtered by age."""
        cutoff = datetime.now(timezone.utc).timestamp() - (max_age_hours * 3600)
        articles = [
            a for a in self._articles.values()
            if a.fetched_at.timestamp() > cutoff
        ]
        articles.sort(key=lambda a: a.fetched_at.timestamp(), reverse=True)
        return articles

    def get_critical_articles(self, max_age_hours: int = 4) -> list[GovArticle]:
        """Get only critical/high priority articles."""
        articles = self.get_articles(max_age_hours=max_age_hours)
        return [a for a in articles if a.priority in ("critical", "high")]

    def _prune_old_articles(self) -> None:
        cutoff = datetime.now(timezone.utc).timestamp() - (72 * 3600)  # 72h retention
        to_remove = [
            aid for aid, a in self._articles.items()
            if a.fetched_at.timestamp() < cutoff
        ]
        for aid in to_remove:
            del self._articles[aid]


def _parse_date(date_str: str) -> Optional[datetime]:
    if not date_str:
        return None
    formats = [
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S %Z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%f%z",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    return None


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text).strip()


# Singleton
gov_rss_service = GovRSSFeedService()
