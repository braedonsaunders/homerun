"""
Sport-Specific Token Classifier

Classifies market tokens into specific sport sub-categories
(ATP Tennis, Soccer, NBA, etc.) for volatility-appropriate execution.

Inspired by terauss .atp_token_categories.json / .ligue1_tokens.json.
Persists classifications to SQL DB instead of JSON files.
"""

from datetime import datetime
from dataclasses import dataclass
from typing import Optional
from sqlalchemy import Column, String, DateTime, Index, select
from models.database import Base, AsyncSessionLocal
from utils.logger import get_logger

logger = get_logger("sport_classifier")


class SportTokenClassification(Base):
    """Persisted sport classification for a token."""

    __tablename__ = "sport_token_classifications"

    token_id = Column(String, primary_key=True)
    sport = Column(String, nullable=False)  # "atp_tennis", "soccer_ligue1", "nba", "nfl", etc.
    sport_category = Column(String, nullable=False)  # "tennis", "soccer", "basketball", "football"
    extra_buffer = Column(String, default="0.01")
    classified_at = Column(DateTime, default=datetime.utcnow)
    source_slug = Column(String, nullable=True)

    __table_args__ = (Index("idx_stc_sport", "sport"),)


# Sport detection patterns: (slug_pattern, sport_id, sport_category, extra_buffer)
SPORT_PATTERNS = [
    ("atp", "atp_tennis", "tennis", 0.01),
    ("wta", "wta_tennis", "tennis", 0.01),
    ("tennis", "tennis_general", "tennis", 0.01),
    ("ligue-1", "soccer_ligue1", "soccer", 0.01),
    ("premier-league", "soccer_epl", "soccer", 0.01),
    ("la-liga", "soccer_laliga", "soccer", 0.01),
    ("champions-league", "soccer_ucl", "soccer", 0.01),
    ("serie-a", "soccer_seriea", "soccer", 0.01),
    ("bundesliga", "soccer_bundesliga", "soccer", 0.01),
    ("mls", "soccer_mls", "soccer", 0.008),
    ("nba", "nba", "basketball", 0.008),
    ("nfl", "nfl", "football", 0.008),
    ("mlb", "mlb", "baseball", 0.005),
    ("nhl", "nhl", "hockey", 0.007),
    ("ufc", "ufc", "mma", 0.01),
    ("boxing", "boxing", "boxing", 0.01),
    ("formula-1", "f1", "motorsport", 0.005),
    ("f1", "f1", "motorsport", 0.005),
]


@dataclass
class SportClassification:
    token_id: str
    sport: str
    sport_category: str
    extra_buffer: float
    is_live_sport: bool  # True if this is a sport that has live in-play trading


class SportClassifier:
    def __init__(self):
        self._cache: dict[str, SportClassification] = {}  # token_id -> classification
        self._loaded = False

    async def load_from_db(self):
        """Load all classifications from DB into memory."""
        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(select(SportTokenClassification))
                for row in result.scalars().all():
                    self._cache[row.token_id] = SportClassification(
                        token_id=row.token_id,
                        sport=row.sport,
                        sport_category=row.sport_category,
                        extra_buffer=float(row.extra_buffer),
                        is_live_sport=row.sport_category in ("tennis", "soccer", "basketball", "mma"),
                    )
            self._loaded = True
            logger.info("Loaded sport classifications", count=len(self._cache))
        except Exception as e:
            logger.error("Failed to load sport classifications", error=str(e))

    def classify_by_slug(self, token_id: str, slug: str) -> Optional[SportClassification]:
        """Classify a token based on its market slug."""
        if token_id in self._cache:
            return self._cache[token_id]

        slug_lower = slug.lower()
        for pattern, sport_id, sport_cat, buffer in SPORT_PATTERNS:
            if pattern in slug_lower:
                classification = SportClassification(
                    token_id=token_id,
                    sport=sport_id,
                    sport_category=sport_cat,
                    extra_buffer=buffer,
                    is_live_sport=sport_cat in ("tennis", "soccer", "basketball", "mma"),
                )
                self._cache[token_id] = classification
                return classification
        return None

    def get_classification(self, token_id: str) -> Optional[SportClassification]:
        return self._cache.get(token_id)

    def get_extra_buffer(self, token_id: str) -> float:
        c = self._cache.get(token_id)
        return c.extra_buffer if c else 0.0

    def is_live_sport(self, token_id: str) -> bool:
        c = self._cache.get(token_id)
        return c.is_live_sport if c else False

    async def persist_classification(self, classification: SportClassification):
        try:
            async with AsyncSessionLocal() as session:
                existing = await session.get(SportTokenClassification, classification.token_id)
                if existing:
                    existing.sport = classification.sport
                    existing.sport_category = classification.sport_category
                    existing.extra_buffer = str(classification.extra_buffer)
                else:
                    session.add(
                        SportTokenClassification(
                            token_id=classification.token_id,
                            sport=classification.sport,
                            sport_category=classification.sport_category,
                            extra_buffer=str(classification.extra_buffer),
                        )
                    )
                await session.commit()
        except Exception as e:
            logger.error("Failed to persist classification", error=str(e))

    def get_stats(self) -> dict:
        by_sport = {}
        for c in self._cache.values():
            by_sport[c.sport] = by_sport.get(c.sport, 0) + 1
        return {"total_classified": len(self._cache), "by_sport": by_sport}


sport_classifier = SportClassifier()
