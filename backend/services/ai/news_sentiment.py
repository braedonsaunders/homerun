"""
News and Sentiment Analyzer.

Searches for recent news related to Polymarket markets and
provides sentiment analysis to inform trading decisions.

Uses httpx to fetch news from free APIs (Google News RSS) and
LLM to summarize and assess sentiment.
"""

from __future__ import annotations

import logging
import urllib.parse
import uuid
from datetime import datetime
from utils.utcnow import utcnow
from typing import Any, Optional
from xml.etree import ElementTree

import httpx

from services.ai import get_llm_manager
from services.ai.llm_provider import LLMMessage
from services.ai.scratchpad import ScratchpadService

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HTTP_TIMEOUT = 10  # seconds
_DEFAULT_MAX_ARTICLES = 5
_USER_AGENT = "Mozilla/5.0 (compatible; Homerun/1.0)"

# ---------------------------------------------------------------------------
# Structured output schema for sentiment analysis
# ---------------------------------------------------------------------------

SENTIMENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "overall_sentiment": {
            "type": "string",
            "enum": ["positive", "negative", "neutral", "mixed"],
            "description": "Overall sentiment direction across all articles",
        },
        "sentiment_score": {
            "type": "number",
            "minimum": -1,
            "maximum": 1,
            "description": "Numeric sentiment score from -1 (very negative) to 1 (very positive)",
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "Confidence in the sentiment assessment (0 to 1)",
        },
        "market_impact": {
            "type": "string",
            "description": "How this news might affect the market outcome probability",
        },
        "key_takeaways": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Key bullet-point takeaways from the news",
        },
        "outcome_implications": {
            "type": "object",
            "description": "How the news affects each possible outcome's likelihood",
        },
    },
    "required": [
        "overall_sentiment",
        "sentiment_score",
        "confidence",
        "market_impact",
        "key_takeaways",
    ],
}


# ---------------------------------------------------------------------------
# NewsSentimentAnalyzer
# ---------------------------------------------------------------------------


class NewsSentimentAnalyzer:
    """
    Searches news and analyzes sentiment for market-relevant topics.

    Uses free news APIs (Google News RSS, etc.) and LLM summarization
    to provide sentiment-aware market intelligence.

    Usage:
        analyzer = NewsSentimentAnalyzer()
        result = await analyzer.search_and_analyze(
            query="Bitcoin price prediction",
            market_context="Will Bitcoin exceed $100K by March 2025?",
        )
    """

    def __init__(self) -> None:
        self._scratchpad = ScratchpadService()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def search_and_analyze(
        self,
        query: str,
        market_context: str = "",
        max_articles: int = _DEFAULT_MAX_ARTICLES,
        model: Optional[str] = None,
    ) -> dict:
        """
        Search for news and analyze sentiment.

        Fetches recent articles matching the query, then uses the LLM to
        assess sentiment and market impact.

        Args:
            query: Search query for relevant news.
            market_context: Optional market question for context.
            max_articles: Maximum number of articles to fetch and analyze.
            model: LLM model to use for sentiment analysis.

        Returns:
            Dict with keys:
                query          -- the search query
                articles       -- list of article dicts
                sentiment      -- overall/score/confidence
                market_impact  -- how news might affect the market
                key_takeaways  -- bullet point summaries
        """
        session_id = uuid.uuid4().hex[:16]

        try:
            # Create a scratchpad session for auditing
            try:
                await self._scratchpad.create_session(
                    session_type="news_sentiment",
                    query=query,
                    model=model,
                )
            except Exception:
                pass  # Scratchpad is optional; don't fail if DB models missing

            # Step 1: Fetch news articles
            articles = await self.fetch_news(query, max_results=max_articles)

            if not articles:
                logger.info("No news articles found for query: %s", query)
                return {
                    "query": query,
                    "articles": [],
                    "sentiment": {
                        "overall": "neutral",
                        "score": 0.0,
                        "confidence": 0.0,
                    },
                    "market_impact": "No recent news found for this topic.",
                    "key_takeaways": ["No relevant news articles were found."],
                    "session_id": session_id,
                }

            # Step 2: Analyze sentiment with LLM
            sentiment_result = await self.analyze_sentiment(
                articles=articles,
                market_question=market_context,
                model=model,
            )

            return {
                "query": query,
                "articles": articles,
                "sentiment": {
                    "overall": sentiment_result.get("overall_sentiment", "neutral"),
                    "score": sentiment_result.get("sentiment_score", 0.0),
                    "confidence": sentiment_result.get("confidence", 0.0),
                },
                "market_impact": sentiment_result.get("market_impact", ""),
                "key_takeaways": sentiment_result.get("key_takeaways", []),
                "outcome_implications": sentiment_result.get(
                    "outcome_implications", {}
                ),
                "session_id": session_id,
            }

        except Exception as exc:
            logger.error(
                "search_and_analyze failed for query '%s': %s",
                query,
                exc,
                exc_info=True,
            )
            return {
                "query": query,
                "articles": [],
                "sentiment": {
                    "overall": "neutral",
                    "score": 0.0,
                    "confidence": 0.0,
                },
                "market_impact": f"Analysis failed: {exc}",
                "key_takeaways": [],
                "session_id": session_id,
                "error": str(exc),
            }

    async def fetch_news(
        self,
        query: str,
        max_results: int = 10,
    ) -> list[dict]:
        """
        Fetch news articles from free sources.

        Uses multiple sources for reliability:
        1. Google News RSS feed (primary -- no API key needed)
        2. Falls back gracefully on failure

        Args:
            query: Search query string.
            max_results: Maximum number of articles to return.

        Returns:
            List of article dicts with keys: title, url, published, source, summary.
        """
        articles: list[dict] = []

        # Source 1: Google News RSS
        google_articles = await self._fetch_google_news_rss(query, max_results)
        articles.extend(google_articles)

        # Source 2: Polymarket blog / community posts (best-effort)
        if len(articles) < max_results:
            poly_articles = await self._fetch_polymarket_news(
                query, max_results - len(articles)
            )
            articles.extend(poly_articles)

        # De-duplicate by URL
        seen_urls: set[str] = set()
        unique_articles: list[dict] = []
        for article in articles:
            url = article.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_articles.append(article)

        return unique_articles[:max_results]

    async def analyze_sentiment(
        self,
        articles: list[dict],
        market_question: str = "",
        model: Optional[str] = None,
    ) -> dict:
        """
        Analyze sentiment from a set of news articles using the LLM.

        Sends article headlines and summaries to the LLM with a structured
        output schema to get a consistent sentiment assessment.

        Args:
            articles: List of article dicts (from fetch_news).
            market_question: Optional market question for context.
            model: LLM model to use.

        Returns:
            Dict conforming to SENTIMENT_SCHEMA with sentiment analysis.
        """
        if not articles:
            return {
                "overall_sentiment": "neutral",
                "sentiment_score": 0.0,
                "confidence": 0.0,
                "market_impact": "No articles to analyze.",
                "key_takeaways": [],
            }

        # Format articles for the LLM
        articles_text = self._format_articles_for_llm(articles)

        # Build the prompt
        system_message = (
            "You are a financial news sentiment analyst specializing in "
            "prediction markets. Analyze the provided news articles and "
            "assess their sentiment and potential market impact.\n\n"
            "Consider:\n"
            "- Overall sentiment direction (positive, negative, neutral, mixed)\n"
            "- Strength and confidence of the sentiment signal\n"
            "- How the news might affect prediction market probabilities\n"
            "- Key takeaways that a trader should know\n"
            "- Implications for specific market outcomes\n\n"
            "Be objective and data-driven. Distinguish between noise and "
            "meaningful signals. Consider both direct and indirect impacts."
        )

        user_content_parts = ["Analyze the sentiment of these news articles:"]
        user_content_parts.append("")
        user_content_parts.append(articles_text)

        if market_question:
            user_content_parts.extend(
                [
                    "",
                    f"MARKET CONTEXT: {market_question}",
                    "",
                    "How do these articles affect the likelihood of the market "
                    "question resolving YES vs NO?",
                ]
            )

        user_content = "\n".join(user_content_parts)

        try:
            manager = get_llm_manager()
            result = await manager.structured_output(
                messages=[
                    LLMMessage(role="system", content=system_message),
                    LLMMessage(role="user", content=user_content),
                ],
                schema=SENTIMENT_SCHEMA,
                model=model,
                purpose="news_sentiment",
            )
            return result

        except Exception as exc:
            logger.error("LLM sentiment analysis failed: %s", exc)
            # Return a safe default rather than crashing
            return {
                "overall_sentiment": "neutral",
                "sentiment_score": 0.0,
                "confidence": 0.0,
                "market_impact": f"Sentiment analysis failed: {exc}",
                "key_takeaways": [
                    f"Found {len(articles)} articles but could not analyze sentiment."
                ],
            }

    async def monitor_market_news(
        self,
        market_id: str,
        market_question: str,
        interval_minutes: int = 30,
    ) -> dict:
        """
        Get latest news for a specific market.

        Convenience method that fetches and analyzes news relevant to a
        particular Polymarket market.

        Args:
            market_id: Polymarket market condition ID.
            market_question: The market's question text.
            interval_minutes: Not used for a single invocation; reserved
                for future periodic monitoring.

        Returns:
            Sentiment analysis result dict.
        """
        # Extract key terms from the market question for a better search
        search_query = self._extract_search_terms(market_question)

        result = await self.search_and_analyze(
            query=search_query,
            market_context=market_question,
        )

        # Attach market metadata
        result["market_id"] = market_id
        result["market_question"] = market_question
        result["monitored_at"] = utcnow().isoformat()

        return result

    # ==================================================================
    # Private: News Source Fetchers
    # ==================================================================

    async def _fetch_google_news_rss(
        self,
        query: str,
        max_results: int = 10,
    ) -> list[dict]:
        """Fetch news articles from Google News RSS feed.

        This is the primary news source as it requires no API key.

        Args:
            query: Search query.
            max_results: Maximum articles to return.

        Returns:
            List of article dicts.
        """
        try:
            encoded_query = urllib.parse.quote(query)
            url = (
                f"https://news.google.com/rss/search"
                f"?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
            )

            async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
                response = await client.get(
                    url,
                    headers={"User-Agent": _USER_AGENT},
                )
                if response.status_code != 200:
                    logger.warning(
                        "Google News RSS returned HTTP %d for query '%s'",
                        response.status_code,
                        query,
                    )
                    return []

                root = ElementTree.fromstring(response.text)
                articles: list[dict] = []

                for item in root.findall(".//item")[:max_results]:
                    title = item.findtext("title", "").strip()
                    link = item.findtext("link", "").strip()
                    pub_date = item.findtext("pubDate", "").strip()
                    source = item.findtext("source", "").strip()

                    # Google News RSS sometimes includes the source in
                    # the title as "Title - Source". Extract it.
                    source_from_title = ""
                    if " - " in title and not source:
                        parts = title.rsplit(" - ", 1)
                        if len(parts) == 2:
                            title = parts[0].strip()
                            source_from_title = parts[1].strip()

                    articles.append(
                        {
                            "title": title,
                            "url": link,
                            "published": pub_date,
                            "source": source or source_from_title,
                            "summary": "",  # RSS doesn't include summaries
                        }
                    )

                logger.debug(
                    "Fetched %d articles from Google News RSS for '%s'",
                    len(articles),
                    query,
                )
                return articles

        except ElementTree.ParseError as exc:
            logger.error("Failed to parse Google News RSS XML: %s", exc)
            return []
        except httpx.TimeoutException:
            logger.warning("Google News RSS request timed out for '%s'", query)
            return []
        except Exception as exc:
            logger.error("Google News RSS fetch failed: %s", exc)
            return []

    async def _fetch_polymarket_news(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[dict]:
        """Fetch Polymarket-specific news by searching for the query on
        Polymarket-adjacent sources.

        This is a best-effort supplementary source. Returns articles
        from crypto/prediction-market focused RSS feeds.

        Args:
            query: Search query.
            max_results: Maximum articles to return.

        Returns:
            List of article dicts.
        """
        articles: list[dict] = []

        try:
            # Search for Polymarket-specific news via Google News
            poly_query = f"Polymarket {query}"
            encoded = urllib.parse.quote(poly_query)
            url = (
                f"https://news.google.com/rss/search"
                f"?q={encoded}&hl=en-US&gl=US&ceid=US:en"
            )

            async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
                response = await client.get(
                    url,
                    headers={"User-Agent": _USER_AGENT},
                )
                if response.status_code != 200:
                    return []

                root = ElementTree.fromstring(response.text)

                for item in root.findall(".//item")[:max_results]:
                    title = item.findtext("title", "").strip()
                    link = item.findtext("link", "").strip()
                    pub_date = item.findtext("pubDate", "").strip()
                    source = item.findtext("source", "").strip()

                    source_from_title = ""
                    if " - " in title and not source:
                        parts = title.rsplit(" - ", 1)
                        if len(parts) == 2:
                            title = parts[0].strip()
                            source_from_title = parts[1].strip()

                    articles.append(
                        {
                            "title": title,
                            "url": link,
                            "published": pub_date,
                            "source": source or source_from_title,
                            "summary": "",
                        }
                    )

        except Exception as exc:
            logger.debug("Polymarket news fetch failed (non-critical): %s", exc)

        return articles

    # ==================================================================
    # Private: Helpers
    # ==================================================================

    @staticmethod
    def _format_articles_for_llm(articles: list[dict]) -> str:
        """Format a list of article dicts into a text block for the LLM.

        Args:
            articles: List of article dicts.

        Returns:
            Formatted text string.
        """
        parts: list[str] = []
        for i, article in enumerate(articles, 1):
            title = article.get("title", "Untitled")
            source = article.get("source", "Unknown source")
            published = article.get("published", "Unknown date")
            summary = article.get("summary", "")

            entry = f"Article {i}:\n  Title: {title}\n  Source: {source}\n  Date: {published}"
            if summary:
                entry += f"\n  Summary: {summary}"
            parts.append(entry)

        return "\n\n".join(parts)

    @staticmethod
    def _extract_search_terms(market_question: str) -> str:
        """Extract meaningful search terms from a market question.

        Strips common prediction-market phrasing like "Will ... by ..."
        to produce a better news search query.

        Args:
            market_question: The full market question text.

        Returns:
            Cleaned search query string.
        """
        q = market_question.strip()

        # Remove common prefixes
        prefixes_to_strip = [
            "Will ",
            "Will the ",
            "Is ",
            "Is the ",
            "Does ",
            "Does the ",
            "Has ",
            "Has the ",
            "Can ",
            "Can the ",
            "Are ",
            "Are the ",
        ]
        for prefix in prefixes_to_strip:
            if q.startswith(prefix):
                q = q[len(prefix) :]
                break

        # Remove trailing question mark
        q = q.rstrip("?").strip()

        # Remove common suffixes like "by March 2025", "before end of 2025"
        # by finding and removing date-like suffixes
        date_markers = [
            " by ",
            " before ",
            " on or before ",
            " prior to ",
            " by the end of ",
            " before the end of ",
        ]
        for marker in date_markers:
            idx = q.lower().find(marker)
            if idx > 0:
                q = q[:idx].strip()
                break

        return q


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

news_sentiment_analyzer = NewsSentimentAnalyzer()
