"""Article clustering for the news workflow.

This groups near-duplicate or same-event articles before market retrieval.
The workflow then runs event extraction and market matching per cluster,
which reduces noisy article-level false positives.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

from services.news.feed_service import NewsArticle

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-'_]{2,}")
_ENTITY_RE = re.compile(r"\b([A-Z][A-Za-z0-9&'.-]+(?:\s+[A-Z][A-Za-z0-9&'.-]+){0,2})\b")

_STOP_WORDS = {
    "about",
    "after",
    "again",
    "against",
    "also",
    "amid",
    "among",
    "another",
    "around",
    "being",
    "below",
    "between",
    "breaking",
    "could",
    "first",
    "from",
    "have",
    "into",
    "its",
    "just",
    "latest",
    "more",
    "most",
    "news",
    "over",
    "said",
    "says",
    "their",
    "there",
    "these",
    "they",
    "this",
    "those",
    "today",
    "update",
    "updates",
    "what",
    "when",
    "where",
    "which",
    "while",
    "will",
    "with",
    "would",
    "year",
    "years",
}

_GENERIC_ENTITY_TERMS = {
    "Associated Press",
    "Bloomberg",
    "Breaking News",
    "Business Insider",
    "CNBC",
    "CNN",
    "Financial Times",
    "Fox News",
    "Google News",
    "New York Times",
    "Reuters",
    "The Guardian",
    "The Wall Street Journal",
    "Washington Post",
}


def _coerce_utc(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _article_ts(article: NewsArticle) -> datetime:
    published = _coerce_utc(article.published)
    if published is not None:
        return published
    return _coerce_utc(article.fetched_at) or datetime.now(timezone.utc)


def _normalize_source(source: str) -> str:
    value = (source or "").strip().lower()
    return value or "unknown"


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a.intersection(b))
    if inter == 0:
        return 0.0
    union = len(a.union(b))
    if union == 0:
        return 0.0
    return inter / union


@dataclass
class _ArticleFeatures:
    article: NewsArticle
    timestamp: datetime
    tokens: set[str]
    focus_tokens: set[str]
    entities: set[str]


@dataclass
class _WorkingCluster:
    members: list[_ArticleFeatures] = field(default_factory=list)
    token_counter: Counter[str] = field(default_factory=Counter)
    focus_counter: Counter[str] = field(default_factory=Counter)
    entity_counter: Counter[str] = field(default_factory=Counter)
    newest_ts: Optional[datetime] = None
    oldest_ts: Optional[datetime] = None

    def add(self, feature: _ArticleFeatures) -> None:
        self.members.append(feature)
        self.token_counter.update(feature.tokens)
        self.focus_counter.update(feature.focus_tokens)
        self.entity_counter.update(feature.entities)
        if self.newest_ts is None or feature.timestamp > self.newest_ts:
            self.newest_ts = feature.timestamp
        if self.oldest_ts is None or feature.timestamp < self.oldest_ts:
            self.oldest_ts = feature.timestamp

    @property
    def token_set(self) -> set[str]:
        return {token for token, _ in self.token_counter.most_common(40)}

    @property
    def focus_set(self) -> set[str]:
        return {token for token, _ in self.focus_counter.most_common(20)}

    @property
    def entity_set(self) -> set[str]:
        return {entity for entity, _ in self.entity_counter.most_common(12)}


@dataclass
class ArticleCluster:
    """Clustered group of related articles."""

    cluster_id: str
    article_key: str
    headline: str
    summary: str
    merged_text: str
    representative: NewsArticle
    articles: list[NewsArticle]
    article_ids: list[str]
    source_keys: set[str]
    source_list: list[str]
    primary_source: str
    primary_url: str
    newest_ts: Optional[datetime]
    oldest_ts: Optional[datetime]

    @property
    def article_count(self) -> int:
        return len(self.article_ids)


class ArticleClusterer:
    """Lightweight semantic-ish clustering using entities and focus tokens."""

    def __init__(
        self,
        *,
        similarity_threshold: float = 0.28,
        recency_window_hours: int = 72,
        max_cluster_size: int = 12,
    ) -> None:
        self._similarity_threshold = similarity_threshold
        self._recency_window = timedelta(hours=max(1, recency_window_hours))
        self._max_cluster_size = max(2, max_cluster_size)

    def cluster(
        self,
        articles: list[NewsArticle],
        *,
        max_clusters: Optional[int] = None,
    ) -> list[ArticleCluster]:
        if not articles:
            return []

        ordered = sorted(articles, key=_article_ts, reverse=True)
        clusters: list[_WorkingCluster] = []
        cluster_cap = max_clusters if max_clusters and max_clusters > 0 else None

        for article in ordered:
            feature = self._build_features(article)
            if not feature.tokens:
                continue

            best_idx = -1
            best_score = 0.0
            for idx, working in enumerate(clusters):
                if len(working.members) >= self._max_cluster_size:
                    continue
                score = self._cluster_similarity(feature, working)
                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx >= 0 and best_score >= self._similarity_threshold:
                clusters[best_idx].add(feature)
                continue

            if cluster_cap is not None and len(clusters) >= cluster_cap:
                continue

            new_cluster = _WorkingCluster()
            new_cluster.add(feature)
            clusters.append(new_cluster)

        finalized = [self._finalize_cluster(c) for c in clusters if c.members]
        finalized.sort(key=lambda c: c.newest_ts or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
        return finalized

    def _cluster_similarity(self, feature: _ArticleFeatures, cluster: _WorkingCluster) -> float:
        if cluster.newest_ts is not None:
            if abs(cluster.newest_ts - feature.timestamp) > self._recency_window:
                return 0.0

        focus_score = _jaccard(feature.focus_tokens, cluster.focus_set)
        entity_score = _jaccard(feature.entities, cluster.entity_set)
        token_score = _jaccard(feature.tokens, cluster.token_set)

        score = (0.55 * focus_score) + (0.30 * entity_score) + (0.15 * token_score)

        # Penalize clusters with no entity agreement unless lexical overlap is strong.
        if feature.entities and cluster.entity_set and entity_score == 0.0 and focus_score < 0.35:
            score *= 0.65
        return score

    def _build_features(self, article: NewsArticle) -> _ArticleFeatures:
        title = (article.title or "").strip()
        summary = (article.summary or "").strip()
        body_text = f"{title}\n{summary}".strip()
        lowered = body_text.lower()

        title_tokens = [
            token for token in _TOKEN_RE.findall(title.lower()) if len(token) > 2 and token not in _STOP_WORDS
        ]
        summary_tokens = [token for token in _TOKEN_RE.findall(lowered) if len(token) > 2 and token not in _STOP_WORDS]

        weighted_counts: Counter[str] = Counter()
        weighted_counts.update(summary_tokens)
        weighted_counts.update(title_tokens)
        weighted_counts.update(title_tokens)  # headline terms matter more

        focus_tokens = {tok for tok, _ in weighted_counts.most_common(18)}
        tokens = set(summary_tokens)

        entities: set[str] = set()
        for entity in _ENTITY_RE.findall(body_text):
            ent = entity.strip()
            if len(ent) < 3:
                continue
            if ent in _GENERIC_ENTITY_TERMS:
                continue
            entities.add(ent.lower())

        return _ArticleFeatures(
            article=article,
            timestamp=_article_ts(article),
            tokens=tokens,
            focus_tokens=focus_tokens,
            entities=entities,
        )

    def _finalize_cluster(self, working: _WorkingCluster) -> ArticleCluster:
        members = sorted(
            working.members,
            key=lambda m: m.timestamp,
            reverse=True,
        )
        representative = members[0].article
        article_ids = [m.article.article_id for m in members]
        cluster_hash = hashlib.sha1("|".join(sorted(article_ids)).encode("utf-8")).hexdigest()[:16]
        headline = (representative.title or "").strip()
        source_keys = {_normalize_source(m.article.source) for m in members}
        source_list = sorted(source_keys)

        summary_chunks = []
        merged_chunks = []
        seen_titles: set[str] = set()
        for m in members[:5]:
            title = (m.article.title or "").strip()
            if not title:
                continue
            dedupe_key = title.lower()
            if dedupe_key in seen_titles:
                continue
            seen_titles.add(dedupe_key)
            summary_chunks.append(title)
            part = title
            if m.article.summary:
                part += f": {m.article.summary.strip()[:220]}"
            merged_chunks.append(part)

        summary = " | ".join(summary_chunks[:4])[:800]
        merged_text = "\n".join(merged_chunks[:6])[:2500]

        return ArticleCluster(
            cluster_id=cluster_hash,
            article_key=f"cluster:{cluster_hash}",
            headline=headline,
            summary=summary,
            merged_text=merged_text,
            representative=representative,
            articles=[m.article for m in members],
            article_ids=article_ids,
            source_keys=source_keys,
            source_list=source_list,
            primary_source=representative.source or (source_list[0] if source_list else ""),
            primary_url=representative.url,
            newest_ts=working.newest_ts,
            oldest_ts=working.oldest_ts,
        )


article_clusterer = ArticleClusterer()
