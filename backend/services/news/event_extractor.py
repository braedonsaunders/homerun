"""
Event Extractor -- LLM structured event extraction from news articles.

Converts raw article text into typed event objects used by the hybrid
retriever to find relevant prediction markets.

Falls back to lightweight keyword extraction when no LLM is available.

Pattern from: OmniEvent (typed event output), Quant-tool (rule-based fallback).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Event types aligned with prediction market categories
# ---------------------------------------------------------------------------

EVENT_TYPES = [
    "policy_change",
    "earnings",
    "election",
    "conflict",
    "regulation",
    "appointment",
    "legal_ruling",
    "economic_data",
    "natural_disaster",
    "product_launch",
    "treaty_agreement",
    "sports_outcome",
    "crypto_market",
    "scientific_breakthrough",
    "other",
]

# Mapping from event type to likely Polymarket categories
EVENT_CATEGORY_AFFINITY: dict[str, list[str]] = {
    "policy_change": ["Politics", "Economics"],
    "earnings": ["Finance", "Economics", "Tech"],
    "election": ["Politics"],
    "conflict": ["Politics", "World"],
    "regulation": ["Politics", "Crypto", "Tech", "Finance"],
    "appointment": ["Politics"],
    "legal_ruling": ["Politics", "Culture"],
    "economic_data": ["Economics", "Finance"],
    "natural_disaster": ["Weather", "World"],
    "product_launch": ["Tech", "Culture"],
    "treaty_agreement": ["Politics", "World"],
    "sports_outcome": ["Sports"],
    "crypto_market": ["Crypto"],
    "scientific_breakthrough": ["Tech", "Science"],
    "other": [],
}


@dataclass
class ExtractedEvent:
    """A typed event extracted from a news article."""

    event_type: str = "other"
    actors: list[str] = field(default_factory=list)
    action: str = ""
    date: Optional[str] = None
    region: Optional[str] = None
    impact_direction: Optional[str] = None  # positive, negative, neutral
    key_entities: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    confidence: float = 0.5

    @property
    def category_affinities(self) -> list[str]:
        """Return prediction-market categories this event is likely related to."""
        return EVENT_CATEGORY_AFFINITY.get(self.event_type, [])

    @property
    def search_terms(self) -> list[str]:
        """Terms to use for keyword-based market search."""
        terms = list(self.key_entities) + list(self.actors) + list(self.keywords)
        if self.action:
            terms.append(self.action)
        return [t for t in terms if t and len(t) > 1]


# ---------------------------------------------------------------------------
# LLM extraction schema
# ---------------------------------------------------------------------------

EVENT_EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "event_type": {
            "type": "string",
            "enum": EVENT_TYPES,
            "description": "The primary type of event described in the article.",
        },
        "actors": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Key people, organisations, or countries involved. "
                "Max 5 entries. Use full names."
            ),
        },
        "action": {
            "type": "string",
            "description": (
                "What happened, in 5-10 words. "
                "E.g. 'announced tariff increase on steel imports'."
            ),
        },
        "date": {
            "type": "string",
            "description": "When the event occurred or will occur (ISO date or natural language).",
        },
        "region": {
            "type": "string",
            "description": "Geographic region most affected (e.g. 'US', 'EU', 'Global').",
        },
        "impact_direction": {
            "type": "string",
            "enum": ["positive", "negative", "neutral", "mixed"],
            "description": "Overall market impact direction of this event.",
        },
        "key_entities": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Specific entities to search for in prediction markets: "
                "company names, bill names, country pairs, crypto tickers, etc. Max 8."
            ),
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "How confident you are in this extraction (0 = guess, 1 = certain).",
        },
    },
    "required": [
        "event_type",
        "actors",
        "action",
        "key_entities",
        "confidence",
    ],
}


# ---------------------------------------------------------------------------
# Keyword-based fallback extractor
# ---------------------------------------------------------------------------

# Simple keyword lists for rule-based event classification
_EVENT_KEYWORDS: dict[str, list[str]] = {
    "election": ["election", "vote", "ballot", "poll", "candidate", "primary", "caucus", "midterm"],
    "policy_change": ["bill", "law", "executive order", "tariff", "sanction", "policy", "regulation"],
    "earnings": ["earnings", "revenue", "quarterly", "profit", "dividend", "eps", "guidance"],
    "conflict": ["war", "military", "troops", "invasion", "strike", "bombing", "ceasefire"],
    "regulation": ["regulate", "sec", "fda", "ftc", "ban", "approve", "compliance", "antitrust"],
    "appointment": ["appoint", "nominate", "resign", "fire", "hire", "ceo", "secretary"],
    "legal_ruling": ["court", "ruling", "verdict", "lawsuit", "supreme court", "judge", "indictment"],
    "economic_data": ["gdp", "inflation", "unemployment", "jobs report", "interest rate", "fed", "cpi"],
    "natural_disaster": ["earthquake", "hurricane", "flood", "wildfire", "tsunami", "storm"],
    "product_launch": ["launch", "release", "unveil", "announce", "iphone", "tesla"],
    "crypto_market": ["bitcoin", "ethereum", "crypto", "blockchain", "defi", "token", "altcoin"],
    "sports_outcome": ["championship", "playoff", "super bowl", "world cup", "finals", "match"],
    "treaty_agreement": ["treaty", "agreement", "deal", "accord", "summit", "g7", "g20", "nato"],
}

# Common entity patterns
_ENTITY_PATTERN = re.compile(
    r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b"  # Multi-word proper nouns
)


def _extract_keywords_fallback(title: str, summary: str) -> ExtractedEvent:
    """Rule-based event extraction when LLM is unavailable."""
    text = f"{title} {summary}".lower()

    # Classify event type
    best_type = "other"
    best_score = 0
    for etype, keywords in _EVENT_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text)
        if score > best_score:
            best_score = score
            best_type = etype

    # Extract proper nouns as entities
    raw_text = f"{title} {summary}"
    entities = list(set(_ENTITY_PATTERN.findall(raw_text)))[:8]

    # Extract significant words as keywords
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "to", "of", "in", "for", "on", "with",
        "at", "by", "from", "as", "into", "and", "but", "or", "not",
        "this", "that", "it", "its", "he", "she", "they", "we", "you",
        "new", "said", "says", "also", "more", "than", "after", "before",
    }
    words = re.findall(r"[a-z]{3,}", text)
    keywords = list(set(w for w in words if w not in stop_words))[:15]

    return ExtractedEvent(
        event_type=best_type,
        actors=[],
        action=title[:80] if title else "",
        key_entities=entities,
        keywords=keywords,
        confidence=0.3 if best_score > 0 else 0.1,
    )


# ---------------------------------------------------------------------------
# Main extractor class
# ---------------------------------------------------------------------------


class EventExtractor:
    """Extracts typed events from news articles using LLM or keyword fallback."""

    async def extract(
        self,
        title: str,
        summary: str = "",
        source: str = "",
        model: Optional[str] = None,
    ) -> ExtractedEvent:
        """Extract a structured event from article text.

        Tries LLM first; falls back to keyword extraction.
        """
        # Try LLM extraction
        event = await self._extract_llm(title, summary, source, model=model)
        if event is not None:
            return event

        # Fallback to keywords
        return _extract_keywords_fallback(title, summary)

    async def extract_batch(
        self,
        articles: list[dict],
        model: Optional[str] = None,
    ) -> list[ExtractedEvent]:
        """Extract events from a batch of articles.

        Each article dict should have 'title' and optionally 'summary', 'source'.
        """
        import asyncio

        sem = asyncio.Semaphore(3)

        async def _one(article: dict) -> ExtractedEvent:
            async with sem:
                return await self.extract(
                    title=article.get("title", ""),
                    summary=article.get("summary", ""),
                    source=article.get("source", ""),
                    model=model,
                )

        return await asyncio.gather(*[_one(a) for a in articles])

    async def _extract_llm(
        self,
        title: str,
        summary: str,
        source: str,
        model: Optional[str] = None,
    ) -> Optional[ExtractedEvent]:
        """Try LLM-based structured extraction."""
        try:
            from services.ai import get_llm_manager
            from services.ai.llm_provider import LLMMessage

            manager = get_llm_manager()
            if not manager.is_available():
                return None
        except Exception:
            return None

        system_prompt = (
            "You are a news event extractor for a prediction market system. "
            "Given a news article, extract the key event as structured data. "
            "Focus on information that would be relevant to prediction markets: "
            "elections, policy changes, economic data, corporate earnings, "
            "geopolitical events, legal rulings, crypto developments, etc.\n\n"
            "Extract the MOST IMPORTANT event from the article. "
            "For key_entities, include specific names that someone would search for "
            "in a prediction market (e.g. 'Donald Trump', 'Federal Reserve', 'Bitcoin')."
        )

        user_prompt = f"TITLE: {title}\n"
        if summary:
            user_prompt += f"SUMMARY: {summary[:500]}\n"
        if source:
            user_prompt += f"SOURCE: {source}\n"

        try:
            result = await manager.structured_output(
                messages=[
                    LLMMessage(role="system", content=system_prompt),
                    LLMMessage(role="user", content=user_prompt),
                ],
                schema=EVENT_EXTRACTION_SCHEMA,
                model=model,
                purpose="news_event_extraction",
            )

            # Build keywords from entities + actors for retrieval
            keywords = []
            for entity in result.get("key_entities", []):
                keywords.extend(entity.lower().split())
            for actor in result.get("actors", []):
                keywords.extend(actor.lower().split())

            return ExtractedEvent(
                event_type=result.get("event_type", "other"),
                actors=result.get("actors", []),
                action=result.get("action", ""),
                date=result.get("date"),
                region=result.get("region"),
                impact_direction=result.get("impact_direction"),
                key_entities=result.get("key_entities", []),
                keywords=list(set(keywords)),
                confidence=result.get("confidence", 0.5),
            )
        except Exception as e:
            logger.debug("LLM event extraction failed: %s", e)
            return None


# Singleton
event_extractor = EventExtractor()
