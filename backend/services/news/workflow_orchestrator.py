"""
Workflow Orchestrator -- Full pipeline coordinator for independent news workflow.

Coordinates the complete article-to-trade-intent pipeline:
  1. Fetch articles (via existing feed_service)
  2. Extract events (event_extractor)
  3. Rebuild/refresh market watcher index
  4. Hybrid retrieval (hybrid_retriever)
  5. LLM reranking (reranker)
  6. Edge estimation (edge_estimator)
  7. Intent generation (intent_generator)
  8. Persist findings + intents to DB
  9. Broadcast WS update

Runs independently of the scanner's strategy system.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Optional

from config import settings

logger = logging.getLogger(__name__)

# Single-thread executor for CPU-bound embedding work
_EMBED_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="news_wf")


class WorkflowOrchestrator:
    """Singleton orchestrator for the independent news workflow pipeline."""

    def __init__(self) -> None:
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_run: Optional[datetime] = None
        self._last_findings_count: int = 0
        self._last_intents_count: int = 0
        self._cycle_count: int = 0
        self._is_cycling = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self, interval_seconds: Optional[int] = None) -> None:
        """Start background workflow loop."""
        if self._running:
            return
        self._running = True
        interval = interval_seconds or settings.NEWS_SCAN_INTERVAL_SECONDS
        self._task = asyncio.create_task(self._loop(interval))
        logger.info("News workflow orchestrator started (interval=%ds)", interval)

    def stop(self) -> None:
        """Stop background loop."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
        logger.info("News workflow orchestrator stopped")

    async def _loop(self, interval: int) -> None:
        """Background loop -- runs a workflow cycle at configured interval."""
        # Wait for initial data to be available
        await asyncio.sleep(10)

        while self._running:
            try:
                if self._is_workflow_enabled():
                    await self.run_cycle()
            except Exception as e:
                logger.error("News workflow cycle error: %s", e)
            await asyncio.sleep(interval)

    def _is_workflow_enabled(self) -> bool:
        """Check if workflow is enabled via config."""
        return getattr(settings, "NEWS_EDGE_ENABLED", True)

    # ------------------------------------------------------------------
    # Main cycle
    # ------------------------------------------------------------------

    async def run_cycle(self) -> dict:
        """Run a single workflow cycle. Can be called manually via API."""
        if self._is_cycling:
            return {"status": "already_running"}

        self._is_cycling = True
        start_time = datetime.now(timezone.utc)

        try:
            from services.news.feed_service import news_feed_service
            from services.news.event_extractor import event_extractor
            from services.news.market_watcher_index import (
                market_watcher_index,
                IndexedMarket,
            )
            from services.news.hybrid_retriever import HybridRetriever
            from services.news.reranker import reranker
            from services.news.edge_estimator import edge_estimator
            from services.news.intent_generator import intent_generator

            # Load workflow settings from DB
            wf_settings = await self._load_settings()

            # ── Step 1: Get articles ────────────────────────────────
            articles = news_feed_service.get_articles(
                max_age_hours=min(wf_settings.get("article_max_age_hours", 6), 48)
            )
            if not articles:
                # Try fetching
                await news_feed_service.fetch_all()
                articles = news_feed_service.get_articles(max_age_hours=6)

            if not articles:
                logger.debug("News workflow: no articles available")
                return {"status": "no_articles", "findings": 0, "intents": 0}

            # Limit articles per cycle for cost control
            articles = articles[:settings.NEWS_MAX_ARTICLES_PER_SCAN]

            # ── Step 2: Extract events ──────────────────────────────
            article_dicts = [
                {
                    "title": a.title,
                    "summary": a.summary or "",
                    "source": a.source,
                }
                for a in articles
            ]
            events = await event_extractor.extract_batch(
                article_dicts, model=wf_settings.get("model")
            )

            # ── Step 3: Rebuild market watcher index ────────────────
            loop = asyncio.get_running_loop()
            market_infos = self._build_market_infos()

            if not market_infos:
                logger.debug("News workflow: no markets available")
                return {"status": "no_markets", "findings": 0, "intents": 0}

            indexed_markets = [
                IndexedMarket(
                    market_id=m["market_id"],
                    question=m["question"],
                    event_title=m.get("event_title", ""),
                    category=m.get("category", ""),
                    yes_price=m.get("yes_price", 0.5),
                    no_price=m.get("no_price", 0.5),
                    liquidity=m.get("liquidity", 0.0),
                    slug=m.get("slug", ""),
                )
                for m in market_infos
            ]

            # Rebuild index in thread pool (CPU-bound)
            if not market_watcher_index._initialized:
                await loop.run_in_executor(_EMBED_EXECUTOR, market_watcher_index.initialize)
            await loop.run_in_executor(
                _EMBED_EXECUTOR, market_watcher_index.rebuild, indexed_markets
            )

            # ── Step 4-6: Retrieve, rerank, estimate for each article ──
            retriever = HybridRetriever(market_watcher_index)
            all_findings = []
            all_intents = []

            top_k = wf_settings.get("top_k", 8)
            rerank_top_n = wf_settings.get("rerank_top_n", 5)
            kw_weight = wf_settings.get("keyword_weight", 0.25)
            sem_weight = wf_settings.get("semantic_weight", 0.45)
            evt_weight = wf_settings.get("event_weight", 0.30)
            sim_threshold = wf_settings.get("similarity_threshold", 0.35)
            min_edge = wf_settings.get("min_edge_percent", 8.0)
            min_conf = wf_settings.get("min_confidence", 0.6)
            model = wf_settings.get("model")

            for article, event in zip(articles, events):
                if event.confidence < 0.1:
                    continue  # Skip very low confidence extractions

                article_text = f"{article.title} {article.summary or ''}"

                # Step 4: Hybrid retrieval
                candidates = retriever.retrieve(
                    event=event,
                    article_text=article_text,
                    top_k=top_k,
                    keyword_weight=kw_weight,
                    semantic_weight=sem_weight,
                    event_weight=evt_weight,
                    similarity_threshold=sim_threshold,
                )

                if not candidates:
                    continue

                # Step 5: LLM rerank
                reranked = await reranker.rerank(
                    article_title=article.title,
                    article_summary=article.summary or "",
                    candidates=candidates,
                    top_n=rerank_top_n,
                    model=model,
                )

                if not reranked:
                    continue

                # Step 6: Edge estimation
                findings = await edge_estimator.estimate_batch(
                    article_title=article.title,
                    article_summary=article.summary or "",
                    article_source=article.source,
                    article_url=article.url,
                    article_id=article.article_id,
                    event=event,
                    reranked=reranked,
                    min_edge_percent=min_edge,
                    min_confidence=min_conf,
                    model=model,
                )

                all_findings.extend(findings)

            # ── Step 7: Generate intents ────────────────────────────
            actionable = [f for f in all_findings if f.actionable]
            auto_trader_min_edge = wf_settings.get("auto_trader_min_edge", 10.0)
            intent_dicts = await intent_generator.generate(
                actionable,
                min_edge=auto_trader_min_edge,
                min_confidence=min_conf,
            )
            all_intents = intent_dicts

            # ── Step 8: Persist to DB ───────────────────────────────
            await self._persist_findings(all_findings)
            await self._persist_intents(all_intents)

            # ── Step 9: Broadcast WS update ─────────────────────────
            try:
                from api.websocket import broadcast_news_update

                await broadcast_news_update(len(all_findings))
            except Exception:
                pass

            # Update stats
            self._last_run = datetime.now(timezone.utc)
            self._last_findings_count = len(all_findings)
            self._last_intents_count = len(all_intents)
            self._cycle_count += 1

            elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
            logger.info(
                "News workflow cycle #%d: %d articles -> %d events -> "
                "%d findings (%d actionable) -> %d intents (%.1fs)",
                self._cycle_count,
                len(articles),
                len(events),
                len(all_findings),
                len(actionable),
                len(all_intents),
                elapsed,
            )

            return {
                "status": "completed",
                "articles": len(articles),
                "events": len(events),
                "findings": len(all_findings),
                "actionable": len(actionable),
                "intents": len(all_intents),
                "elapsed_seconds": round(elapsed, 2),
            }

        except Exception as e:
            logger.error("News workflow cycle failed: %s", e, exc_info=True)
            return {"status": "error", "error": str(e)}
        finally:
            self._is_cycling = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_market_infos(self) -> list[dict]:
        """Build market info list from the scanner's cached universe."""
        try:
            from services import scanner as scanner_inst

            cached_markets = getattr(scanner_inst, "_cached_markets", [])
            cached_events = getattr(scanner_inst, "_cached_events", [])
            cached_prices = getattr(scanner_inst, "_cached_prices", {})

            if not cached_markets:
                return []

            # Build event lookup
            event_by_market: dict[str, object] = {}
            for ev in cached_events:
                for m in ev.markets:
                    event_by_market[m.id] = ev

            infos = []
            seen: set[str] = set()
            for m in cached_markets:
                if m.id in seen:
                    continue
                seen.add(m.id)

                yes_price = 0.5
                no_price = 0.5
                if m.outcome_prices and len(m.outcome_prices) >= 2:
                    yes_price = m.outcome_prices[0]
                    no_price = m.outcome_prices[1]
                elif m.clob_token_ids:
                    for i, tid in enumerate(m.clob_token_ids):
                        p = cached_prices.get(tid)
                        if p is not None:
                            price_val = p if isinstance(p, (int, float)) else p.get("mid", 0.5)
                            if i == 0:
                                yes_price = price_val
                            else:
                                no_price = price_val

                ev = event_by_market.get(m.id)
                infos.append({
                    "market_id": m.id,
                    "question": m.question,
                    "event_title": ev.title if ev else "",
                    "category": ev.category or "" if ev else "",
                    "yes_price": yes_price,
                    "no_price": no_price,
                    "liquidity": m.liquidity,
                    "slug": m.slug,
                })

            return infos
        except Exception as e:
            logger.error("Failed to build market infos: %s", e)
            return []

    async def _load_settings(self) -> dict:
        """Load workflow settings from DB AppSettings."""
        try:
            from models.database import AsyncSessionLocal, AppSettings
            from sqlalchemy import select

            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(AppSettings).where(AppSettings.id == "default")
                )
                db = result.scalar_one_or_none()
                if not db:
                    return {}

                return {
                    "top_k": getattr(db, "news_workflow_top_k", 8) or 8,
                    "rerank_top_n": getattr(db, "news_workflow_rerank_top_n", 5) or 5,
                    "similarity_threshold": getattr(db, "news_workflow_similarity_threshold", 0.35) or 0.35,
                    "keyword_weight": getattr(db, "news_workflow_keyword_weight", 0.25) or 0.25,
                    "semantic_weight": getattr(db, "news_workflow_semantic_weight", 0.45) or 0.45,
                    "event_weight": getattr(db, "news_workflow_event_weight", 0.30) or 0.30,
                    "min_edge_percent": getattr(db, "news_workflow_min_edge_percent", 8.0) or 8.0,
                    "min_confidence": getattr(db, "news_workflow_min_confidence", 0.6) or 0.6,
                    "auto_trader_min_edge": getattr(db, "news_workflow_auto_trader_min_edge", 10.0) or 10.0,
                    "model": getattr(db, "news_workflow_model", None),
                    "article_max_age_hours": 6,
                }
        except Exception as e:
            logger.warning("Failed to load workflow settings: %s", e)
            return {}

    async def _persist_findings(self, findings: list) -> int:
        """Persist findings to DB."""
        if not findings:
            return 0
        try:
            from models.database import AsyncSessionLocal, NewsWorkflowFinding
            from sqlalchemy.dialects.sqlite import insert as sqlite_insert

            async with AsyncSessionLocal() as session:
                for f in findings:
                    stmt = sqlite_insert(NewsWorkflowFinding).values(
                        id=f.id,
                        article_id=f.article_id,
                        market_id=f.market_id,
                        article_title=f.article_title,
                        article_source=f.article_source,
                        article_url=f.article_url,
                        market_question=f.market_question,
                        market_price=f.market_price,
                        model_probability=f.model_probability,
                        edge_percent=f.edge_percent,
                        direction=f.direction,
                        confidence=f.confidence,
                        retrieval_score=f.retrieval_score,
                        semantic_score=f.semantic_score,
                        keyword_score=f.keyword_score,
                        event_score=f.event_score,
                        rerank_score=f.rerank_score,
                        event_graph=f.event_graph,
                        evidence=f.evidence,
                        reasoning=f.reasoning,
                        actionable=f.actionable,
                        created_at=f.created_at,
                    ).on_conflict_do_nothing()
                    await session.execute(stmt)
                await session.commit()
            return len(findings)
        except Exception as e:
            logger.warning("Failed to persist findings: %s", e)
            return 0

    async def _persist_intents(self, intents: list[dict]) -> int:
        """Persist trade intents to DB."""
        if not intents:
            return 0
        try:
            from models.database import AsyncSessionLocal, NewsTradeIntent
            from sqlalchemy.dialects.sqlite import insert as sqlite_insert

            async with AsyncSessionLocal() as session:
                for intent in intents:
                    stmt = sqlite_insert(NewsTradeIntent).values(**intent).on_conflict_do_nothing()
                    await session.execute(stmt)
                await session.commit()
            return len(intents)
        except Exception as e:
            logger.warning("Failed to persist intents: %s", e)
            return 0

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        return {
            "running": self._running,
            "is_cycling": self._is_cycling,
            "cycle_count": self._cycle_count,
            "last_run": self._last_run.isoformat() if self._last_run else None,
            "last_findings_count": self._last_findings_count,
            "last_intents_count": self._last_intents_count,
        }


# Singleton
workflow_orchestrator = WorkflowOrchestrator()
