"""
Semantic matching engine for news-to-market matching.

Embeds news articles and market questions into a shared vector space
using sentence-transformers, then uses FAISS for fast similarity search.

Supports a lightweight fallback (TF-IDF + cosine similarity) when
sentence-transformers/FAISS are not installed, so the system degrades
gracefully.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np

from config import settings
from services.news.feed_service import NewsArticle

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try to import optional ML dependencies
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
    # Disable tokenizer parallelism to avoid segfaults when called from
    # multiple threads (e.g. asyncio.to_thread in news_edge + API routes).
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
class MarketInfo:
    """Lightweight market descriptor for the matcher."""

    market_id: str
    question: str
    event_title: str = ""
    category: str = ""
    yes_price: float = 0.5
    no_price: float = 0.5
    liquidity: float = 0.0
    slug: str = ""
    end_date: Optional[str] = None

    # Set after embedding
    embedding: Optional[np.ndarray] = None


@dataclass
class NewsMarketMatch:
    """A matched pair: news article + market with similarity score."""

    article: NewsArticle
    market: MarketInfo
    similarity: float
    match_method: str = "semantic"  # "semantic" or "tfidf_fallback"


# ---------------------------------------------------------------------------
# Semantic Matcher
# ---------------------------------------------------------------------------

# Default model: small, fast, runs on CPU, 384 dimensions
_DEFAULT_MODEL = "all-MiniLM-L6-v2"


class SemanticMatcher:
    """
    Matches news articles to prediction markets using vector similarity.

    Uses sentence-transformers for embedding and FAISS for fast search.
    Falls back to TF-IDF cosine similarity when ML deps aren't available.
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        self._model_name = model_name
        self._model: Optional[SentenceTransformer] = None
        self._initialized = False
        self._lock = threading.Lock()

        # Market index
        self._markets: list[MarketInfo] = []
        self._market_embeddings: Optional[np.ndarray] = None
        self._faiss_index: Optional[object] = None  # faiss.IndexFlatIP

        # Cache-hit tracking: skip redundant re-embedding when data hasn't changed
        self._market_index_hash: Optional[str] = None
        self._last_embed_count: int = 0  # articles embedded in last pass

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self) -> bool:
        """Load the embedding model. Returns True if ML mode is available."""
        with self._lock:
            if self._initialized:
                return self._model is not None
            if _HAS_TRANSFORMERS:
                try:
                    # Force local-cache/offline model loading so DNS/network
                    # failures never block startup.
                    os.environ.setdefault("HF_HUB_OFFLINE", "1")
                    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
                    # Force CPU to avoid MPS/CUDA segfaults in threaded context
                    device = os.environ.get("EMBEDDING_DEVICE", "cpu")
                    self._model = SentenceTransformer(self._model_name, device=device)
                    # Smoke-test: encode a tiny string to verify native code works
                    _test = self._model.encode(["test"], show_progress_bar=False, normalize_embeddings=True)
                    if _test is None or len(_test) == 0:
                        raise RuntimeError("Model encode returned empty result")
                    self._initialized = True
                    logger.info("Semantic matcher initialized with model '%s'", self._model_name)
                    return True
                except Exception as e:
                    logger.warning(
                        "Failed to load sentence-transformers model '%s': %s. Falling back to TF-IDF.",
                        self._model_name,
                        e,
                    )
                    self._model = None
            else:
                logger.info("sentence-transformers not installed. Using TF-IDF fallback for news matching.")

            self._initialized = True
            return False

    @property
    def is_ml_mode(self) -> bool:
        """Whether the full ML pipeline is available."""
        return self._model is not None

    def get_model(self) -> Optional["SentenceTransformer"]:
        """Return the loaded SentenceTransformer model (or None).

        Other modules (e.g. MarketWatcherIndex) should call this to share
        the single model instance instead of loading a duplicate.
        """
        if not self._initialized:
            self.initialize()
        return self._model

    # ------------------------------------------------------------------
    # Market index management
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_market_hash(markets: list[MarketInfo]) -> str:
        """Fast content hash of the market list for change detection."""
        h = hashlib.md5(usedforsecurity=False)
        for m in markets:
            h.update(m.market_id.encode())
            h.update(m.question.encode())
        return h.hexdigest()

    def update_market_index(self, markets: list[MarketInfo]) -> int:
        """Rebuild the market embedding index.

        Call this after fetching new market data from the scanner.
        Skips re-embedding when the market list is unchanged.
        Returns the number of markets indexed.
        """
        if not self._initialized:
            self.initialize()

        if not markets:
            with self._lock:
                self._markets = markets
                self._market_embeddings = None
                self._faiss_index = None
                self._market_index_hash = None
            return 0

        # Skip expensive re-embedding when the market set hasn't changed
        new_hash = self._compute_market_hash(markets)
        if new_hash == self._market_index_hash and self._market_embeddings is not None:
            # Market IDs/questions unchanged — just update the MarketInfo list
            # (prices may have changed, but embeddings don't depend on prices)
            with self._lock:
                self._markets = markets
            logger.debug("Market index unchanged (%d markets), skipping re-embed", len(markets))
            return len(markets)

        texts = [self._market_to_text(m) for m in markets]

        with self._lock:
            self._markets = markets

            if self._model is not None:
                try:
                    embeddings = self._model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
                    self._market_embeddings = np.array(embeddings, dtype=np.float32)
                except Exception as e:
                    logger.warning("Market embedding failed, disabling ML mode: %s", e)
                    self._model = None
                    self._market_embeddings = None
                    self._faiss_index = None
                    self._market_index_hash = None
                    return len(markets)

                if _HAS_FAISS and self._market_embeddings.ndim == 2:
                    try:
                        embs = self._market_embeddings
                        if not embs.flags["C_CONTIGUOUS"]:
                            embs = np.ascontiguousarray(embs, dtype=np.float32)
                            self._market_embeddings = embs
                        dim = embs.shape[1]
                        self._faiss_index = faiss.IndexFlatIP(dim)
                        self._faiss_index.add(embs)
                    except Exception as e:
                        logger.warning("FAISS index build failed, using numpy fallback: %s", e)
                        self._faiss_index = None
                else:
                    self._faiss_index = None
            else:
                self._market_embeddings = None
                self._faiss_index = None

            self._market_index_hash = new_hash

        logger.debug("Market index updated: %d markets", len(markets))
        return len(markets)

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def embed_articles(self, articles: list[NewsArticle]) -> int:
        """Embed articles that don't have embeddings yet.

        Returns number of newly embedded articles.
        """
        if not self._model:
            return 0

        unembedded = [a for a in articles if a.embedding is None]
        if not unembedded:
            return 0

        texts = [self._article_to_text(a) for a in unembedded]
        with self._lock:
            try:
                embeddings = self._model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
            except Exception as e:
                logger.warning("Article embedding failed, disabling ML mode: %s", e)
                self._model = None
                return 0

        for article, emb in zip(unembedded, embeddings):
            article.embedding = emb.tolist()

        return len(unembedded)

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def match_articles_to_markets(
        self,
        articles: list[NewsArticle],
        top_k: int = 3,
        threshold: Optional[float] = None,
    ) -> list[NewsMarketMatch]:
        """Find the best market matches for each article.

        Args:
            articles: Articles to match (must be embedded if ML mode).
            top_k: Max markets to return per article.
            threshold: Minimum similarity. Defaults to config value.

        Returns:
            List of NewsMarketMatch sorted by similarity descending.
        """
        if not self._initialized:
            self.initialize()

        if threshold is None:
            threshold = settings.NEWS_SIMILARITY_THRESHOLD

        if not self._markets:
            return []

        with self._lock:
            if self._model is not None and self._market_embeddings is not None:
                return self._match_semantic(articles, top_k, threshold)
            else:
                return self._match_tfidf(articles, top_k, threshold)

    def find_matches(
        self,
        query: str,
        top_k: int = 10,
        threshold: Optional[float] = None,
    ) -> list[dict]:
        """Find cached news articles semantically similar to *query*.

        This is the StrategySDK entry-point.  It embeds *query* on-the-fly
        and searches the in-memory article store for the closest matches.
        Articles must have been embedded by a prior ``embed_articles`` call
        (done automatically by the scanner prefetch).

        Args:
            query: Free-text search string (e.g. a market question).
            top_k: Maximum number of results.
            threshold: Minimum similarity score.  Defaults to config value.

        Returns:
            List of dicts: title, source, score, published_at, summary.
        """
        if not self._initialized:
            self.initialize()

        if threshold is None:
            threshold = settings.NEWS_SIMILARITY_THRESHOLD

        if not query.strip():
            return []

        # Gather embedded articles from the feed store
        try:
            from services.news.feed_service import news_feed_service

            articles = news_feed_service.get_articles(
                max_age_hours=getattr(settings, "NEWS_ARTICLE_TTL_HOURS", 168),
            )
        except Exception:
            articles = []

        embedded = [a for a in articles if a.embedding is not None]

        if self._model is not None and embedded:
            # Semantic path: embed query, dot-product against article embeddings
            with self._lock:
                try:
                    q_emb = self._model.encode(
                        [query],
                        show_progress_bar=False,
                        normalize_embeddings=True,
                    )
                    q_vec = np.array(q_emb, dtype=np.float32)
                    if q_vec.ndim == 1:
                        q_vec = q_vec.reshape(1, -1)
                except Exception as e:
                    logger.warning("Query embedding failed in find_matches: %s", e)
                    return []

            try:
                art_embs = np.array([a.embedding for a in embedded], dtype=np.float32)
            except (ValueError, TypeError):
                return []
            if art_embs.ndim != 2 or q_vec.shape[1] != art_embs.shape[1]:
                return []

            sim = (q_vec @ art_embs.T).flatten()
            top_indices = np.argsort(sim)[::-1][:top_k]
            results = []
            for idx in top_indices:
                score = float(sim[int(idx)])
                if score < threshold:
                    continue
                a = embedded[int(idx)]
                results.append(
                    {
                        "title": a.title,
                        "source": a.source or "",
                        "score": score,
                        "published_at": str(a.published_at or a.fetched_at or ""),
                        "summary": a.summary or "",
                    }
                )
            return results

        # TF-IDF fallback (works even without ML deps)
        if not articles:
            return []
        article_texts = [self._article_to_text(a) for a in articles]
        vocab = _build_vocab(article_texts + [query])
        if not vocab:
            return []
        query_vec = _tfidf_vector(query, vocab)
        q_norm = np.linalg.norm(query_vec)
        if q_norm == 0:
            return []
        scored: list[tuple[int, float]] = []
        for i, a in enumerate(articles):
            a_vec = _tfidf_vector(article_texts[i], vocab)
            a_norm = np.linalg.norm(a_vec)
            if a_norm == 0:
                continue
            score = float(np.dot(query_vec, a_vec) / (q_norm * a_norm))
            if score >= threshold:
                scored.append((i, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            {
                "title": articles[i].title,
                "source": articles[i].source or "",
                "score": s,
                "published_at": str(articles[i].published_at or articles[i].fetched_at or ""),
                "summary": articles[i].summary or "",
            }
            for i, s in scored[:top_k]
        ]

    def _match_semantic(
        self,
        articles: list[NewsArticle],
        top_k: int,
        threshold: float,
    ) -> list[NewsMarketMatch]:
        """Match using sentence embeddings + FAISS/numpy."""
        matches: list[NewsMarketMatch] = []

        # Collect articles that have embeddings
        embedded_articles = [a for a in articles if a.embedding is not None]
        if not embedded_articles:
            return []

        try:
            article_embs = np.array([a.embedding for a in embedded_articles], dtype=np.float32)
        except (ValueError, TypeError) as e:
            logger.warning("Failed to build article embedding array: %s", e)
            return []

        # Validate array shape (must be 2-D with correct embedding dim)
        if article_embs.ndim != 2:
            logger.warning(
                "Article embeddings have unexpected shape %s, skipping",
                article_embs.shape,
            )
            return []

        if self._faiss_index is not None and self._market_embeddings is not None:
            # Validate dimensions match the FAISS index
            index_dim = self._market_embeddings.shape[1]
            if article_embs.shape[1] != index_dim:
                logger.warning(
                    "Embedding dim mismatch: articles=%d, index=%d",
                    article_embs.shape[1],
                    index_dim,
                )
                return []

            # Ensure C-contiguous layout for FAISS
            if not article_embs.flags["C_CONTIGUOUS"]:
                article_embs = np.ascontiguousarray(article_embs, dtype=np.float32)

            k = min(top_k, len(self._markets))
            if k < 1:
                return []

            try:
                scores, indices = self._faiss_index.search(article_embs, k)
            except Exception as e:
                logger.warning("FAISS search failed: %s", e)
                return []

            for i, article in enumerate(embedded_articles):
                for j in range(k):
                    idx = int(indices[i][j])
                    score = float(scores[i][j])
                    if idx < 0 or idx >= len(self._markets) or score < threshold:
                        continue
                    matches.append(
                        NewsMarketMatch(
                            article=article,
                            market=self._markets[idx],
                            similarity=score,
                            match_method="semantic",
                        )
                    )
        elif self._market_embeddings is not None:
            # Fallback: numpy dot product (still uses embeddings)
            if article_embs.shape[1] != self._market_embeddings.shape[1]:
                return []
            sim_matrix = article_embs @ self._market_embeddings.T

            for i, article in enumerate(embedded_articles):
                top_indices = np.argsort(sim_matrix[i])[::-1][:top_k]
                for idx in top_indices:
                    score = float(sim_matrix[i][idx])
                    if score < threshold:
                        continue
                    matches.append(
                        NewsMarketMatch(
                            article=article,
                            market=self._markets[idx],
                            similarity=score,
                            match_method="semantic",
                        )
                    )

        matches.sort(key=lambda m: m.similarity, reverse=True)
        return matches

    def _match_tfidf(
        self,
        articles: list[NewsArticle],
        top_k: int,
        threshold: float,
    ) -> list[NewsMarketMatch]:
        """Fallback matching using TF-IDF cosine similarity (no ML deps)."""
        matches: list[NewsMarketMatch] = []

        article_texts = [self._article_to_text(a) for a in articles]
        market_texts = [self._market_to_text(m) for m in self._markets]

        # Build simple word-frequency vectors
        all_texts = article_texts + market_texts
        vocab = _build_vocab(all_texts)

        if not vocab:
            return []

        article_vecs = [_tfidf_vector(t, vocab) for t in article_texts]
        market_vecs = [_tfidf_vector(t, vocab) for t in market_texts]

        for i, article in enumerate(articles):
            a_vec = article_vecs[i]
            a_norm = np.linalg.norm(a_vec)
            if a_norm == 0:
                continue

            scores = []
            for j, m_vec in enumerate(market_vecs):
                m_norm = np.linalg.norm(m_vec)
                if m_norm == 0:
                    continue
                score = float(np.dot(a_vec, m_vec) / (a_norm * m_norm))
                scores.append((j, score))

            scores.sort(key=lambda x: x[1], reverse=True)

            for idx, score in scores[:top_k]:
                if score < threshold:
                    continue
                matches.append(
                    NewsMarketMatch(
                        article=article,
                        market=self._markets[idx],
                        similarity=score,
                        match_method="tfidf_fallback",
                    )
                )

        matches.sort(key=lambda m: m.similarity, reverse=True)
        return matches

    # ------------------------------------------------------------------
    # Text preparation
    # ------------------------------------------------------------------

    @staticmethod
    def _article_to_text(article: NewsArticle) -> str:
        """Convert article to searchable text."""
        parts = [article.title]
        if article.summary:
            parts.append(article.summary)
        if article.category:
            parts.append(article.category)
        return " ".join(parts)

    @staticmethod
    def _market_to_text(market: MarketInfo) -> str:
        """Convert market info to searchable text."""
        parts = [market.question]
        if market.event_title:
            parts.append(market.event_title)
        if market.category:
            parts.append(market.category)
        return " ".join(parts)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        return {
            "initialized": self._initialized,
            "ml_mode": self.is_ml_mode,
            "model": self._model_name if self.is_ml_mode else "tfidf_fallback",
            "markets_indexed": len(self._markets),
            "has_faiss": _HAS_FAISS,
            "market_index_cached": self._market_index_hash is not None,
        }


# ======================================================================
# TF-IDF helpers (no external deps)
# ======================================================================

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
    "need",
    "dare",
    "ought",
    "used",
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
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "between",
    "out",
    "off",
    "over",
    "under",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
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
    "his",
    "her",
    "their",
    "what",
    "which",
    "who",
    "whom",
}


def _tokenize(text: str) -> list[str]:
    """Simple tokenization: lowercase, alphanumeric words, remove stop words."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in _STOP_WORDS and len(w) > 1]


def _build_vocab(texts: list[str], max_features: int = 5000) -> dict[str, int]:
    """Build a vocabulary from texts, returning word -> index mapping."""
    word_counts: dict[str, int] = {}
    for text in texts:
        for word in set(_tokenize(text)):
            word_counts[word] = word_counts.get(word, 0) + 1

    # Sort by frequency, take top features
    sorted_words = sorted(word_counts, key=word_counts.get, reverse=True)
    return {word: i for i, word in enumerate(sorted_words[:max_features])}


def _tfidf_vector(text: str, vocab: dict[str, int]) -> np.ndarray:
    """Create a simple TF vector for text given vocabulary."""
    vec = np.zeros(len(vocab), dtype=np.float32)
    tokens = _tokenize(text)
    for token in tokens:
        if token in vocab:
            vec[vocab[token]] += 1.0
    # Normalize
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


# ======================================================================
# Singleton
# ======================================================================

semantic_matcher = SemanticMatcher()
