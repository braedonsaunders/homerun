"""
Market Watcher Index -- Reverse index for Option D (market-first matching).

Pre-indexes all active prediction markets with keywords + embeddings so
incoming articles can be matched quickly without re-embedding the entire
market universe each time.

Pattern from: Polymarket Agents (Chroma vector DB), Quant-tool (pgvector).
"""

from __future__ import annotations

import logging
import os
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try optional ML deps (same approach as semantic_matcher.py)
# ---------------------------------------------------------------------------

_HAS_TRANSFORMERS = False
_HAS_FAISS = False
# FAISS remains enabled by default; set NEWS_ENABLE_FAISS=0 only for emergency fallback.
_ENABLE_FAISS = os.environ.get("NEWS_ENABLE_FAISS", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}

try:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from sentence_transformers import SentenceTransformer

    _HAS_TRANSFORMERS = True
except ImportError:
    SentenceTransformer = None  # type: ignore

if _ENABLE_FAISS:
    try:
        import faiss

        try:
            faiss_threads = int(os.environ.get("NEWS_FAISS_THREADS", "1"))
            if hasattr(faiss, "omp_set_num_threads"):
                faiss.omp_set_num_threads(max(1, faiss_threads))
        except Exception:
            # Keep FAISS available even if thread pinning is unsupported.
            pass

        _HAS_FAISS = True
    except ImportError:
        faiss = None  # type: ignore
else:
    faiss = None  # type: ignore


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class IndexedMarket:
    """A market in the watcher index."""

    market_id: str
    question: str
    event_title: str = ""
    category: str = ""
    yes_price: float = 0.5
    no_price: float = 0.5
    liquidity: float = 0.0
    slug: str = ""
    end_date: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None


@dataclass
class SearchResult:
    """A market search result with scored breakdown."""

    market: IndexedMarket
    keyword_score: float = 0.0
    semantic_score: float = 0.0
    combined_score: float = 0.0


# ---------------------------------------------------------------------------
# BM25-like keyword scorer
# ---------------------------------------------------------------------------

_STOP_WORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "shall",
    "can",
    "to",
    "of",
    "in",
    "for",
    "on",
    "with",
    "at",
    "by",
    "from",
    "as",
    "into",
    "and",
    "but",
    "or",
    "if",
    "while",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "he",
    "she",
    "they",
    "them",
    "what",
    "which",
    "who",
    "whom",
    "not",
    "no",
    "yes",
    "will",
    "about",
    "up",
    "out",
    "than",
    "more",
    "very",
    "so",
}


def _tokenize(text: str) -> list[str]:
    """Lowercase tokenization with stop-word removal."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in _STOP_WORDS and len(w) > 1]


def _keyword_match_score(query_tokens: list[str], doc_tokens: list[str]) -> float:
    """Simple BM25-inspired keyword overlap score."""
    if not query_tokens or not doc_tokens:
        return 0.0
    doc_set = set(doc_tokens)
    hits = sum(1 for t in query_tokens if t in doc_set)
    # Normalize by query length, with document length penalty
    score = hits / max(len(query_tokens), 1)
    # Slight IDF-like boost for rare term hits
    if len(doc_tokens) > 0:
        score *= min(1.0, 10.0 / max(len(doc_tokens), 1))
    return min(score, 1.0)


# ---------------------------------------------------------------------------
# Market Watcher Index
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = "all-MiniLM-L6-v2"


class MarketWatcherIndex:
    """
    Reverse index of active prediction markets.

    Supports:
    - Keyword search (BM25-like)
    - Semantic search (FAISS / numpy cosine)
    - Category/metadata filtering
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        self._model_name = model_name
        self._model: Optional[SentenceTransformer] = None
        self._initialized = False
        self._lock = threading.Lock()

        self._markets: list[IndexedMarket] = []
        self._market_tokens: list[list[str]] = []  # tokenized text per market
        self._embeddings: Optional[np.ndarray] = None
        self._faiss_index: Optional[object] = None
        self._last_rebuild: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self) -> bool:
        """Load embedding model. Returns True if ML mode available."""
        with self._lock:
            if self._initialized:
                return self._model is not None
            if _HAS_TRANSFORMERS:
                try:
                    # Avoid startup hangs when DNS/network is unavailable.
                    os.environ.setdefault("HF_HUB_OFFLINE", "1")
                    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
                    device = os.environ.get("EMBEDDING_DEVICE", "cpu")
                    self._model = SentenceTransformer(self._model_name, device=device)
                    self._model.encode(["test"], show_progress_bar=False, normalize_embeddings=True)
                    self._initialized = True
                    logger.info("Market watcher index initialized (ML mode)")
                    return True
                except Exception as e:
                    logger.warning("Failed to load embedding model: %s", e)
                    self._model = None
            self._initialized = True
            return False

    @property
    def is_ml_mode(self) -> bool:
        return self._model is not None

    @property
    def market_count(self) -> int:
        return len(self._markets)

    # ------------------------------------------------------------------
    # Rebuild index from scanner markets
    # ------------------------------------------------------------------

    def rebuild(self, markets: list[IndexedMarket]) -> int:
        """Rebuild the entire index from a list of markets."""
        if not self._initialized:
            self.initialize()

        # Tokenize for keyword search
        tokens_list = []
        for m in markets:
            tags_text = " ".join(m.tags or [])
            text = f"{m.question} {m.event_title} {m.category} {tags_text} {m.slug}"
            m.keywords = _tokenize(text)
            tokens_list.append(m.keywords)

        # Embed for semantic search
        if self._model is not None and markets:
            texts = [f"{m.question} {m.event_title} {m.category} {' '.join(m.tags or [])} {m.slug}" for m in markets]
            try:
                embs = self._model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
                embs = np.array(embs, dtype=np.float32)

                with self._lock:
                    self._embeddings = embs

                    if _HAS_FAISS and embs.ndim == 2:
                        if not embs.flags["C_CONTIGUOUS"]:
                            embs = np.ascontiguousarray(embs, dtype=np.float32)
                            self._embeddings = embs
                        idx = faiss.IndexFlatIP(embs.shape[1])
                        idx.add(embs)
                        self._faiss_index = idx
                    else:
                        self._faiss_index = None

                    for market, emb in zip(markets, embs):
                        market.embedding = emb
            except Exception as e:
                logger.warning("Market embedding failed: %s", e)
                self._embeddings = None
                self._faiss_index = None
        else:
            with self._lock:
                self._embeddings = None
                self._faiss_index = None

        with self._lock:
            self._markets = markets
            self._market_tokens = tokens_list
            self._last_rebuild = datetime.now(timezone.utc)

        logger.info("Market watcher index rebuilt: %d markets", len(markets))
        return len(markets)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_terms: list[str],
        query_embedding: Optional[np.ndarray] = None,
        category_filter: Optional[list[str]] = None,
        min_liquidity: float = 0.0,
        top_k: int = 20,
        keyword_weight: float = 0.25,
        semantic_weight: float = 0.45,
    ) -> list[SearchResult]:
        """Hybrid search: keyword + semantic + metadata filters.

        Args:
            query_terms: Tokenized search terms (from event extractor).
            query_embedding: Pre-computed embedding of article text.
            category_filter: Only return markets in these categories.
            min_liquidity: Minimum liquidity filter.
            top_k: Max results to return.
            keyword_weight: Weight for keyword match score.
            semantic_weight: Weight for semantic similarity.

        Returns:
            Top-K SearchResult objects sorted by combined_score desc.
        """
        if not self._markets:
            return []

        results: list[SearchResult] = []

        # Pre-compute semantic scores if we have embeddings
        semantic_scores: Optional[np.ndarray] = None
        if query_embedding is not None and self._embeddings is not None and self._model is not None:
            qe = np.array(query_embedding, dtype=np.float32)
            if qe.ndim == 1:
                qe = qe.reshape(1, -1)

            with self._lock:
                if self._faiss_index is not None:
                    try:
                        if not qe.flags["C_CONTIGUOUS"]:
                            qe = np.ascontiguousarray(qe, dtype=np.float32)
                        k = min(top_k * 3, len(self._markets))
                        scores, indices = self._faiss_index.search(qe, k)
                        semantic_scores = np.zeros(len(self._markets), dtype=np.float32)
                        for j in range(k):
                            idx = int(indices[0][j])
                            if 0 <= idx < len(self._markets):
                                semantic_scores[idx] = float(scores[0][j])
                    except Exception:
                        semantic_scores = None

                if semantic_scores is None and self._embeddings is not None:
                    # Numpy fallback
                    if qe.shape[1] == self._embeddings.shape[1]:
                        semantic_scores = (qe @ self._embeddings.T).flatten()

        # Score each market
        for i, market in enumerate(self._markets):
            # Metadata filter
            if min_liquidity > 0 and market.liquidity < min_liquidity:
                continue
            if category_filter:
                if market.category and market.category not in category_filter:
                    continue

            # Keyword score
            kw_score = _keyword_match_score(query_terms, self._market_tokens[i])

            # Semantic score
            sem_score = 0.0
            if semantic_scores is not None:
                sem_score = float(max(semantic_scores[i], 0.0))

            # Combined
            combined = keyword_weight * kw_score + semantic_weight * sem_score

            if combined > 0.01:
                results.append(
                    SearchResult(
                        market=market,
                        keyword_score=kw_score,
                        semantic_score=sem_score,
                        combined_score=combined,
                    )
                )

        results.sort(key=lambda r: r.combined_score, reverse=True)
        return results[:top_k]

    def embed_text(self, text: str) -> Optional[np.ndarray]:
        """Embed a single text string for search queries."""
        if self._model is None:
            return None
        try:
            emb = self._model.encode([text], show_progress_bar=False, normalize_embeddings=True)
            return emb[0]
        except Exception:
            return None

    def get_status(self) -> dict:
        return {
            "initialized": self._initialized,
            "ml_mode": self.is_ml_mode,
            "market_count": len(self._markets),
            "has_faiss": _HAS_FAISS and self._faiss_index is not None,
            "last_rebuild": self._last_rebuild.isoformat() if self._last_rebuild else None,
        }


# Singleton
market_watcher_index = MarketWatcherIndex()
