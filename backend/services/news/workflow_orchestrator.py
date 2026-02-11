"""News workflow orchestrator.

Worker-safe, DB-first pipeline:
  1) Fetch + sync articles
  2) Extract events (LLM/fallback)
  3) Rebuild market watcher index from scanner snapshot markets
  4) Hybrid retrieval
  5) Optional LLM reranking (adaptive)
  6) Optional LLM edge estimation (budget-guarded)
  7) Intent generation + persistence
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from models.database import LLMUsageLog, NewsTradeIntent, NewsWorkflowFinding
from services.news import shared_state

logger = logging.getLogger(__name__)

# Single-thread executor for CPU-bound embedding/index work.
_EMBED_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="news_wf")


@dataclass
class CycleBudget:
    """LLM budget guardrails for one workflow cycle.

    This reuses the global LLM usage/accounting and adds per-cycle/per-hour caps.
    """

    llm_available: bool
    global_spend_remaining_usd: float
    cycle_spend_cap_usd: float
    hourly_spend_cap_usd: float
    hourly_news_spend_usd: float
    cycle_llm_call_cap: int
    estimated_cost_per_call_usd: float = 0.02
    llm_calls_used: int = 0
    llm_calls_skipped: int = 0
    estimated_cycle_spend_used_usd: float = 0.0

    def reserve_calls(self, requested_calls: int) -> int:
        """Reserve up to requested LLM calls under current guardrails."""
        allowed = 0
        for _ in range(max(0, requested_calls)):
            if not self.llm_available:
                break
            if self.llm_calls_used >= self.cycle_llm_call_cap:
                break
            if self.estimated_cycle_spend_used_usd + self.estimated_cost_per_call_usd > self.cycle_spend_cap_usd:
                break
            if self.hourly_news_spend_usd + self.estimated_cost_per_call_usd > self.hourly_spend_cap_usd:
                break
            if self.global_spend_remaining_usd < self.estimated_cost_per_call_usd:
                break

            self.llm_calls_used += 1
            self.estimated_cycle_spend_used_usd += self.estimated_cost_per_call_usd
            self.hourly_news_spend_usd += self.estimated_cost_per_call_usd
            self.global_spend_remaining_usd = max(
                0.0, self.global_spend_remaining_usd - self.estimated_cost_per_call_usd
            )
            allowed += 1

        skipped = max(0, requested_calls - allowed)
        if skipped:
            self.llm_calls_skipped += skipped
        return allowed

    @property
    def degraded_mode(self) -> bool:
        return (not self.llm_available) or self.llm_calls_skipped > 0

    def remaining_budget_usd(self) -> float:
        return round(
            max(
                0.0,
                min(
                    self.global_spend_remaining_usd,
                    self.cycle_spend_cap_usd - self.estimated_cycle_spend_used_usd,
                    self.hourly_spend_cap_usd - self.hourly_news_spend_usd,
                ),
            ),
            6,
        )


class WorkflowOrchestrator:
    """Worker-safe orchestrator for the independent news workflow."""

    def __init__(self) -> None:
        self._last_run: Optional[datetime] = None
        self._last_findings_count: int = 0
        self._last_intents_count: int = 0
        self._cycle_count: int = 0
        self._is_cycling = False

    async def run_cycle(self, session: AsyncSession) -> dict:
        """Run one cycle. Caller (worker) owns loop scheduling and control flow."""
        if self._is_cycling:
            return {"status": "already_running"}

        self._is_cycling = True
        started_at = datetime.now(timezone.utc)

        try:
            from services.news.edge_estimator import WorkflowFinding, edge_estimator
            from services.news.event_extractor import event_extractor
            from services.news.feed_service import news_feed_service
            from services.news.hybrid_retriever import HybridRetriever
            from services.news.intent_generator import intent_generator
            from services.news.market_watcher_index import IndexedMarket, market_watcher_index
            from services.news.reranker import reranker

            wf_settings = await shared_state.get_news_settings(session)

            # 1) Sync articles from provider feeds.
            try:
                fetched = await news_feed_service.fetch_all()
                if fetched:
                    await news_feed_service.persist_to_db()
                    await news_feed_service.prune_db()
            except Exception as exc:
                logger.warning("News fetch sync failed (continuing with cache): %s", exc)

            articles = news_feed_service.get_articles(
                max_age_hours=min(wf_settings.get("article_max_age_hours", 6), 48)
            )
            articles.sort(key=lambda a: a.fetched_at.timestamp(), reverse=True)
            articles = articles[: settings.NEWS_MAX_ARTICLES_PER_SCAN]
            if not articles:
                return {
                    "status": "no_articles",
                    "findings": 0,
                    "intents": 0,
                    "stats": {
                        "articles": 0,
                        "events": 0,
                        "market_count": 0,
                        "llm_calls_used": 0,
                        "llm_calls_skipped": 0,
                    },
                }

            # 2) Market universe from scanner snapshot (DB source of truth).
            market_infos = await self._build_market_infos(session)
            if not market_infos:
                return {
                    "status": "no_markets",
                    "findings": 0,
                    "intents": 0,
                    "stats": {
                        "articles": len(articles),
                        "events": 0,
                        "market_count": 0,
                        "llm_calls_used": 0,
                        "llm_calls_skipped": 0,
                    },
                }

            # 3) Build/refresh watcher index.
            indexed_markets = [
                IndexedMarket(
                    market_id=m["market_id"],
                    question=m["question"],
                    event_title=m.get("event_title", ""),
                    category=m.get("category", ""),
                    yes_price=float(m.get("yes_price", 0.5) or 0.5),
                    no_price=float(m.get("no_price", 0.5) or 0.5),
                    liquidity=float(m.get("liquidity", 0.0) or 0.0),
                    slug=m.get("slug", ""),
                )
                for m in market_infos
            ]

            loop = asyncio.get_running_loop()
            if not market_watcher_index._initialized:
                await loop.run_in_executor(_EMBED_EXECUTOR, market_watcher_index.initialize)
            await loop.run_in_executor(_EMBED_EXECUTOR, market_watcher_index.rebuild, indexed_markets)

            # 4) Budget guardrails (global LLM accounting + cycle/hour caps).
            llm_manager = None
            usage = {}
            try:
                from services.ai import get_llm_manager

                llm_manager = get_llm_manager()
                if llm_manager.is_available():
                    usage = await llm_manager.get_usage_stats()
            except Exception:
                llm_manager = None
            global_remaining = float(usage.get("spend_remaining_usd", 0.0) or 0.0)
            hourly_news_spend = await self._hourly_news_spend_usd(session)
            budget = CycleBudget(
                llm_available=bool(llm_manager and llm_manager.is_available() and global_remaining > 0),
                global_spend_remaining_usd=global_remaining,
                cycle_spend_cap_usd=float(wf_settings.get("cycle_spend_cap_usd", 0.25) or 0.25),
                hourly_spend_cap_usd=float(wf_settings.get("hourly_spend_cap_usd", 2.0) or 2.0),
                hourly_news_spend_usd=hourly_news_spend,
                cycle_llm_call_cap=int(wf_settings.get("cycle_llm_call_cap", 30) or 30),
            )

            # 5) Event extraction with adaptive LLM usage.
            events = []
            for article in articles:
                allow_llm = budget.reserve_calls(1) == 1
                event = await event_extractor.extract(
                    title=article.title,
                    summary=article.summary or "",
                    source=article.source,
                    model=wf_settings.get("model"),
                    allow_llm=allow_llm,
                )
                events.append(event)

            retriever = HybridRetriever(market_watcher_index)
            market_metadata_by_id = {
                m["market_id"]: {
                    "id": m["market_id"],
                    "slug": m.get("slug"),
                    "event_slug": m.get("event_slug"),
                    "event_title": m.get("event_title"),
                    "liquidity": m.get("liquidity", 0.0),
                    "yes_price": m.get("yes_price", 0.5),
                    "no_price": m.get("no_price", 0.5),
                    "token_ids": m.get("token_ids") or [],
                }
                for m in market_infos
            }

            top_k = int(wf_settings.get("top_k", 8) or 8)
            rerank_top_n = int(wf_settings.get("rerank_top_n", 5) or 5)
            kw_weight = float(wf_settings.get("keyword_weight", 0.25) or 0.25)
            sem_weight = float(wf_settings.get("semantic_weight", 0.45) or 0.45)
            evt_weight = float(wf_settings.get("event_weight", 0.30) or 0.30)
            sim_threshold = float(wf_settings.get("similarity_threshold", 0.35) or 0.35)
            min_edge = float(wf_settings.get("min_edge_percent", 8.0) or 8.0)
            min_conf = float(wf_settings.get("min_confidence", 0.6) or 0.6)
            max_edge_evals_per_article = int(
                wf_settings.get("max_edge_evals_per_article", 3) or 3
            )
            cache_ttl_minutes = int(wf_settings.get("cache_ttl_minutes", 30) or 30)

            all_findings: list[WorkflowFinding] = []
            market_sources_seen: dict[str, set[str]] = defaultdict(set)

            for article, event in zip(articles, events):
                if event.confidence < 0.1:
                    continue

                article_text = f"{article.title} {article.summary or ''}".strip()
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

                use_llm_rerank = self._should_use_llm_rerank(candidates)
                allow_llm_rerank = use_llm_rerank and budget.reserve_calls(1) == 1
                reranked = await reranker.rerank(
                    article_title=article.title,
                    article_summary=article.summary or "",
                    candidates=candidates,
                    top_n=rerank_top_n,
                    model=wf_settings.get("model"),
                    allow_llm=allow_llm_rerank,
                )
                if not reranked:
                    continue

                reranked = [
                    r for r in reranked if r.rerank_score >= max(0.2, sim_threshold * 0.7)
                ]
                if not reranked:
                    continue
                reranked = reranked[: max(1, max_edge_evals_per_article)]

                # Source-diversity gate for expensive per-market edge calls.
                diversity_gated = []
                for rc in reranked:
                    seen = market_sources_seen.get(rc.market_id, set())
                    src = (article.source or "").strip().lower()
                    if src and src in seen:
                        budget.llm_calls_skipped += 1
                        continue
                    diversity_gated.append(rc)
                if not diversity_gated:
                    continue

                # Reuse recent cached findings (article+market+price bucket) before LLM.
                cache_keys = [
                    self._cache_key(article.article_id, rc.market_id, rc.candidate.yes_price)
                    for rc in diversity_gated
                ]
                cached = await self._load_cached_findings(
                    session,
                    cache_keys=cache_keys,
                    ttl_minutes=cache_ttl_minutes,
                )

                cached_hits: list[WorkflowFinding] = []
                to_estimate = []
                for rc in diversity_gated:
                    cache_key = self._cache_key(
                        article.article_id,
                        rc.market_id,
                        rc.candidate.yes_price,
                    )
                    row = cached.get(cache_key)
                    if row is not None:
                        cached_hits.append(self._row_to_finding(row))
                    else:
                        to_estimate.append(rc)

                llm_calls_for_edges = budget.reserve_calls(len(to_estimate))
                findings = await edge_estimator.estimate_batch(
                    article_title=article.title,
                    article_summary=article.summary or "",
                    article_source=article.source,
                    article_url=article.url,
                    article_id=article.article_id,
                    event=event,
                    reranked=to_estimate,
                    min_edge_percent=min_edge,
                    min_confidence=min_conf,
                    model=wf_settings.get("model"),
                    allow_llm=llm_calls_for_edges > 0,
                    max_llm_calls=llm_calls_for_edges,
                )

                article_findings = cached_hits + findings
                for finding in article_findings:
                    self._assign_finding_keys(finding)
                    market_sources_seen[finding.market_id].add(
                        (finding.article_source or "").strip().lower()
                    )
                all_findings.extend(article_findings)

            deduped_findings = self._dedupe_findings(all_findings)

            if bool(wf_settings.get("require_second_source", False)):
                by_market_sources: dict[str, set[str]] = defaultdict(set)
                for f in deduped_findings:
                    src = (f.article_source or "").strip().lower()
                    if src:
                        by_market_sources[f.market_id].add(src)
                for f in deduped_findings:
                    if len(by_market_sources.get(f.market_id, set())) < 2:
                        f.actionable = False

            actionable = [f for f in deduped_findings if f.actionable]
            intents: list[dict] = []
            if bool(wf_settings.get("auto_trader_enabled", True)):
                intents = await intent_generator.generate(
                    actionable,
                    min_edge=float(wf_settings.get("auto_trader_min_edge", 10.0) or 10.0),
                    min_confidence=min_conf,
                    market_metadata_by_id=market_metadata_by_id,
                )

            await self._persist_findings(session, deduped_findings)
            await self._persist_intents(session, intents)

            self._cycle_count += 1
            self._last_run = datetime.now(timezone.utc)
            self._last_findings_count = len(deduped_findings)
            self._last_intents_count = len(intents)

            elapsed = (datetime.now(timezone.utc) - started_at).total_seconds()
            stats = {
                "cycle_count": self._cycle_count,
                "articles": len(articles),
                "events": len(events),
                "market_count": len(market_infos),
                "findings": len(deduped_findings),
                "actionable": len(actionable),
                "intents": len(intents),
                "llm_calls_used": budget.llm_calls_used,
                "llm_calls_skipped": budget.llm_calls_skipped,
                "elapsed_seconds": round(elapsed, 2),
                "market_index": {
                    "initialized": market_watcher_index._initialized,
                    "ml_mode": market_watcher_index.is_ml_mode,
                    "market_count": market_watcher_index.market_count,
                    "has_faiss": market_watcher_index.get_status().get("has_faiss", False),
                    "last_rebuild": market_watcher_index.get_status().get("last_rebuild"),
                },
            }

            logger.info(
                "News workflow cycle #%d: %d articles -> %d findings (%d actionable) -> %d intents (%.1fs)",
                self._cycle_count,
                len(articles),
                len(deduped_findings),
                len(actionable),
                len(intents),
                elapsed,
            )

            return {
                "status": "completed",
                "articles": len(articles),
                "events": len(events),
                "findings": len(deduped_findings),
                "actionable": len(actionable),
                "intents": len(intents),
                "elapsed_seconds": round(elapsed, 2),
                "degraded_mode": budget.degraded_mode,
                "budget_remaining": budget.remaining_budget_usd(),
                "stats": stats,
            }

        except Exception as exc:
            logger.error("News workflow cycle failed: %s", exc, exc_info=True)
            return {
                "status": "error",
                "error": str(exc),
                "degraded_mode": True,
                "budget_remaining": 0.0,
                "stats": {
                    "cycle_count": self._cycle_count,
                    "llm_calls_used": 0,
                    "llm_calls_skipped": 0,
                },
            }
        finally:
            self._is_cycling = False

    async def _build_market_infos(self, session: AsyncSession) -> list[dict]:
        """Build market info from scanner DB snapshot opportunities."""
        from services import shared_state as scanner_state

        opportunities = await scanner_state.get_opportunities_from_db(session, None)
        if not opportunities:
            return []

        infos: list[dict] = []
        seen: set[str] = set()

        for opp in opportunities:
            event_slug = getattr(opp, "event_slug", None)
            for m in opp.markets:
                market_id = str(m.get("id") or "").strip()
                if not market_id or market_id in seen:
                    continue
                seen.add(market_id)

                # Try to infer token IDs from market payload and opportunity positions.
                token_ids = []
                raw_token_ids = m.get("clob_token_ids")
                if isinstance(raw_token_ids, list):
                    token_ids = [str(t) for t in raw_token_ids if t]
                if not token_ids:
                    yes_token = None
                    no_token = None
                    for pos in getattr(opp, "positions_to_take", []) or []:
                        if str(pos.get("market_id") or m.get("id") or "") != market_id:
                            continue
                        outcome = str(pos.get("outcome") or "").upper()
                        tid = pos.get("token_id")
                        if not tid:
                            continue
                        if outcome == "YES":
                            yes_token = tid
                        elif outcome == "NO":
                            no_token = tid
                    if yes_token or no_token:
                        token_ids = [t for t in [yes_token, no_token] if t]

                infos.append(
                    {
                        "market_id": market_id,
                        "question": str(m.get("question") or ""),
                        "event_title": str(getattr(opp, "event_title", "") or ""),
                        "event_slug": event_slug,
                        "category": str(getattr(opp, "category", "") or ""),
                        "yes_price": float(m.get("yes_price", 0.5) or 0.5),
                        "no_price": float(m.get("no_price", 0.5) or 0.5),
                        "liquidity": float(m.get("liquidity", 0.0) or 0.0),
                        "slug": m.get("slug"),
                        "token_ids": token_ids,
                    }
                )

        return infos

    async def _hourly_news_spend_usd(self, session: AsyncSession) -> float:
        cutoff = datetime.utcnow() - timedelta(hours=1)
        result = await session.execute(
            select(func.coalesce(func.sum(LLMUsageLog.cost_usd), 0.0)).where(
                LLMUsageLog.requested_at >= cutoff,
                LLMUsageLog.success == True,  # noqa: E712
                LLMUsageLog.purpose.like("news%"),
            )
        )
        return float(result.scalar() or 0.0)

    @staticmethod
    def _should_use_llm_rerank(candidates) -> bool:
        """Use LLM rerank only when retrieval confidence is ambiguous."""
        if not candidates:
            return False
        top = candidates[0].combined_score
        second = candidates[1].combined_score if len(candidates) > 1 else 0.0
        # High-confidence obvious match: skip expensive rerank.
        if top >= 0.82 and (top - second) >= 0.18:
            return False
        # Very weak matches: skip rerank and drop later.
        if top < 0.15:
            return False
        return True

    @staticmethod
    def _cache_key(article_id: str, market_id: str, market_price: float) -> str:
        price_bucket = int(round(float(market_price or 0.0) * 1000))
        raw = f"{article_id}:{market_id}:{price_bucket}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

    @staticmethod
    def _signal_key(
        article_id: str,
        market_id: str,
        direction: str,
        market_price: float,
        model_probability: float,
    ) -> str:
        mkt_bucket = int(round(float(market_price or 0.0) * 1000))
        mdl_bucket = int(round(float(model_probability or 0.0) * 1000))
        raw = f"{article_id}:{market_id}:{direction}:{mkt_bucket}:{mdl_bucket}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

    def _assign_finding_keys(self, finding) -> None:
        direction = finding.direction or "buy_yes"
        cache_key = self._cache_key(
            finding.article_id,
            finding.market_id,
            float(finding.market_price or 0.0),
        )
        signal_key = self._signal_key(
            finding.article_id,
            finding.market_id,
            direction,
            float(finding.market_price or 0.0),
            float(finding.model_probability or 0.0),
        )
        finding.cache_key = cache_key
        finding.signal_key = signal_key
        finding.id = signal_key[:16]

    def _dedupe_findings(self, findings):
        by_signal = {}
        for f in findings:
            key = getattr(f, "signal_key", None) or getattr(f, "id", None)
            if not key:
                self._assign_finding_keys(f)
                key = getattr(f, "signal_key", None) or f.id
            existing = by_signal.get(key)
            if existing is None:
                by_signal[key] = f
                continue
            if (f.edge_percent, f.confidence) > (existing.edge_percent, existing.confidence):
                by_signal[key] = f
        return list(by_signal.values())

    async def _load_cached_findings(
        self,
        session: AsyncSession,
        cache_keys: list[str],
        ttl_minutes: int,
    ) -> dict[str, NewsWorkflowFinding]:
        if not cache_keys:
            return {}
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=max(1, ttl_minutes))
        result = await session.execute(
            select(NewsWorkflowFinding)
            .where(NewsWorkflowFinding.cache_key.in_(cache_keys))
            .where(NewsWorkflowFinding.created_at >= cutoff)
            .order_by(NewsWorkflowFinding.created_at.desc())
        )
        rows = result.scalars().all()
        out: dict[str, NewsWorkflowFinding] = {}
        for row in rows:
            key = row.cache_key
            if not key or key in out:
                continue
            out[key] = row
        return out

    @staticmethod
    def _row_to_finding(row: NewsWorkflowFinding):
        from services.news.edge_estimator import WorkflowFinding

        return WorkflowFinding(
            id=row.id,
            article_id=row.article_id,
            market_id=row.market_id,
            article_title=row.article_title,
            article_source=row.article_source or "",
            article_url=row.article_url or "",
            market_question=row.market_question,
            market_price=float(row.market_price or 0.5),
            model_probability=float(row.model_probability or 0.5),
            edge_percent=float(row.edge_percent or 0.0),
            direction=row.direction or "buy_yes",
            confidence=float(row.confidence or 0.0),
            retrieval_score=float(row.retrieval_score or 0.0),
            semantic_score=float(row.semantic_score or 0.0),
            keyword_score=float(row.keyword_score or 0.0),
            event_score=float(row.event_score or 0.0),
            rerank_score=float(row.rerank_score or 0.0),
            event_graph=row.event_graph or {},
            evidence=row.evidence or {},
            reasoning=row.reasoning or "",
            actionable=bool(row.actionable),
            created_at=row.created_at,
            signal_key=row.signal_key,
            cache_key=row.cache_key,
        )

    async def _persist_findings(self, session: AsyncSession, findings: list) -> int:
        if not findings:
            return 0
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert

        count = 0
        for f in findings:
            stmt = sqlite_insert(NewsWorkflowFinding).values(
                id=f.id,
                article_id=f.article_id,
                market_id=f.market_id,
                article_title=f.article_title,
                article_source=f.article_source,
                article_url=f.article_url,
                signal_key=getattr(f, "signal_key", None),
                cache_key=getattr(f, "cache_key", None),
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
            ).on_conflict_do_update(
                index_elements=["id"],
                set_={
                    "signal_key": getattr(f, "signal_key", None),
                    "cache_key": getattr(f, "cache_key", None),
                    "market_price": f.market_price,
                    "model_probability": f.model_probability,
                    "edge_percent": f.edge_percent,
                    "direction": f.direction,
                    "confidence": f.confidence,
                    "retrieval_score": f.retrieval_score,
                    "semantic_score": f.semantic_score,
                    "keyword_score": f.keyword_score,
                    "event_score": f.event_score,
                    "rerank_score": f.rerank_score,
                    "event_graph": f.event_graph,
                    "evidence": f.evidence,
                    "reasoning": f.reasoning,
                    "actionable": f.actionable,
                    "created_at": f.created_at,
                },
            )
            await session.execute(stmt)
            count += 1
        await session.commit()
        return count

    async def _persist_intents(self, session: AsyncSession, intents: list[dict]) -> int:
        if not intents:
            return 0

        count = 0
        for intent in intents:
            signal_key = intent.get("signal_key")
            query = select(NewsTradeIntent)
            if signal_key:
                query = query.where(NewsTradeIntent.signal_key == signal_key)
            else:
                query = query.where(NewsTradeIntent.id == intent["id"])
            existing_result = await session.execute(
                query
            )
            existing = existing_result.scalar_one_or_none()
            if existing is None:
                session.add(NewsTradeIntent(**intent))
                count += 1
                continue

            # Preserve consumed outcomes and only refresh pending/submitted rows.
            if existing.status in {"pending", "submitted"}:
                for key, value in intent.items():
                    setattr(existing, key, value)
                count += 1

        await session.commit()
        return count

    def get_status(self) -> dict:
        return {
            "is_cycling": self._is_cycling,
            "cycle_count": self._cycle_count,
            "last_run": self._last_run.isoformat() if self._last_run else None,
            "last_findings_count": self._last_findings_count,
            "last_intents_count": self._last_intents_count,
        }


workflow_orchestrator = WorkflowOrchestrator()
