"""News workflow orchestrator.

Worker-safe, DB-first pipeline:
  1) Fetch + sync articles
  2) Extract events (LLM/fallback)
  3) Rebuild market watcher index from active market metadata
  4) Hybrid retrieval
  5) Optional LLM reranking (adaptive)
  6) Optional LLM edge estimation (budget-guarded)
  7) Intent generation + persistence
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from utils.utcnow import utcnow
from typing import Any, Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from models.database import LLMUsageLog, NewsTradeIntent, NewsWorkflowFinding
from services.news import shared_state

logger = logging.getLogger(__name__)

# Single-thread executor for CPU-bound embedding/index work.
_EMBED_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="news_wf")
_MAX_WORKFLOW_CYCLE_SECONDS = 90.0


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
            from services.news.article_clusterer import article_clusterer
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
            articles.sort(
                key=lambda a: (
                    self._coerce_datetime(getattr(a, "published", None))
                    or self._coerce_datetime(getattr(a, "fetched_at", None))
                    or datetime.min.replace(tzinfo=timezone.utc)
                ),
                reverse=True,
            )
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

            cycle_llm_call_cap = int(wf_settings.get("cycle_llm_call_cap", 30) or 30)
            cluster_limit = max(
                8,
                min(
                    len(articles),
                    max(12, int(cycle_llm_call_cap * 2)),
                ),
            )
            clusters = article_clusterer.cluster(articles, max_clusters=None)
            if len(clusters) > cluster_limit:
                def _cluster_rank(cluster):
                    newest = self._coerce_datetime(getattr(cluster, "newest_ts", None))
                    recency = newest.timestamp() if newest else 0.0
                    source_count = len(getattr(cluster, "source_list", []) or [])
                    article_count = int(getattr(cluster, "article_count", 0) or 0)
                    # Prioritize clusters with corroboration across outlets, then size, then recency.
                    return (
                        1 if source_count >= 2 else 0,
                        article_count,
                        recency,
                    )

                clusters = sorted(clusters, key=_cluster_rank, reverse=True)[:cluster_limit]
                clusters.sort(
                    key=lambda c: (
                        self._coerce_datetime(getattr(c, "newest_ts", None))
                        or datetime.min.replace(tzinfo=timezone.utc)
                    ),
                    reverse=True,
                )
            if not clusters:
                return {
                    "status": "no_clusters",
                    "findings": 0,
                    "intents": 0,
                    "stats": {
                        "articles": len(articles),
                        "clusters": 0,
                        "events": 0,
                        "market_count": 0,
                        "llm_calls_used": 0,
                        "llm_calls_skipped": 0,
                    },
                }

            market_min_liquidity = float(wf_settings.get("market_min_liquidity", 500.0) or 500.0)
            market_max_days_to_resolution = int(wf_settings.get("market_max_days_to_resolution", 365) or 365)
            # 2) Market universe from live markets first, scanner snapshot fallback.
            market_infos = await self._build_market_infos(
                session,
                min_liquidity=market_min_liquidity,
                max_days_to_resolution=market_max_days_to_resolution,
            )
            if not market_infos:
                return {
                    "status": "no_markets",
                    "findings": 0,
                    "intents": 0,
                    "stats": {
                        "articles": len(articles),
                        "clusters": len(clusters),
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
                    end_date=m.get("end_date"),
                    tags=list(m.get("tags") or []),
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
            llm_available = bool(llm_manager and llm_manager.is_available())
            local_model_mode = self._is_local_model_mode(
                model=wf_settings.get("model"),
                usage=usage,
            )

            if local_model_mode and llm_available:
                global_remaining = float("inf")
                cycle_spend_cap = 10_000.0
                hourly_spend_cap = 10_000.0
                hourly_news_spend = 0.0
                effective_call_cap = max(
                    cycle_llm_call_cap,
                    max(80, len(clusters) * 3),
                )
                estimated_cost_per_call = 0.0
            else:
                if "spend_remaining_usd" in usage:
                    spend_limit = float(usage.get("spend_limit_usd", 0.0) or 0.0)
                    if spend_limit <= 0:
                        global_remaining = float("inf")
                    else:
                        global_remaining = float(usage.get("spend_remaining_usd", 0.0) or 0.0)
                else:
                    # If usage read fails but provider is up, don't hard-disable LLM.
                    global_remaining = float("inf") if llm_available else 0.0
                hourly_news_spend = await self._hourly_news_spend_usd(session)
                cycle_spend_cap = float(wf_settings.get("cycle_spend_cap_usd", 0.25) or 0.25)
                hourly_spend_cap = float(wf_settings.get("hourly_spend_cap_usd", 2.0) or 2.0)
                effective_call_cap = cycle_llm_call_cap
                estimated_cost_per_call = 0.02

            budget = CycleBudget(
                llm_available=bool(llm_available and global_remaining > 0),
                global_spend_remaining_usd=global_remaining,
                cycle_spend_cap_usd=cycle_spend_cap,
                hourly_spend_cap_usd=hourly_spend_cap,
                hourly_news_spend_usd=hourly_news_spend,
                cycle_llm_call_cap=effective_call_cap,
                estimated_cost_per_call_usd=estimated_cost_per_call,
            )

            # 5) Event extraction with adaptive LLM usage.
            # Local models get broader quotas; remote models stay capped.
            if local_model_mode and budget.llm_available:
                event_llm_quota = len(clusters)
                rerank_llm_quota = len(clusters)
            else:
                event_llm_quota = (
                    max(0, min(len(clusters), int(effective_call_cap * 0.25))) if effective_call_cap >= 5 else 0
                )
                rerank_llm_quota = (
                    max(0, min(len(clusters), int(effective_call_cap * 0.3))) if effective_call_cap >= 5 else 0
                )
            event_llm_used = 0
            rerank_llm_used = 0
            alignment_dropped = 0
            cluster_events = 0

            retriever = HybridRetriever(market_watcher_index)
            market_metadata_by_id = {
                m["market_id"]: {
                    "id": m["market_id"],
                    "slug": m.get("slug"),
                    "event_slug": m.get("event_slug"),
                    "event_ticker": m.get("event_ticker"),
                    "platform": m.get("platform"),
                    "event_title": m.get("event_title"),
                    "liquidity": m.get("liquidity", 0.0),
                    "yes_price": m.get("yes_price", 0.5),
                    "no_price": m.get("no_price", 0.5),
                    "end_date": m.get("end_date"),
                    "tags": m.get("tags") or [],
                    "token_ids": m.get("token_ids") or [],
                }
                for m in market_infos
            }

            top_k = int(wf_settings.get("top_k", 8) or 8)
            rerank_top_n = int(wf_settings.get("rerank_top_n", 5) or 5)
            kw_weight = float(wf_settings.get("keyword_weight", 0.25) or 0.25)
            sem_weight = float(wf_settings.get("semantic_weight", 0.45) or 0.45)
            evt_weight = float(wf_settings.get("event_weight", 0.30) or 0.30)
            sim_threshold = float(wf_settings.get("similarity_threshold", 0.42) or 0.42)
            min_keyword_signal = float(wf_settings.get("min_keyword_signal", 0.04) or 0.04)
            min_semantic_signal = float(wf_settings.get("min_semantic_signal", 0.22) or 0.22)
            min_edge = float(wf_settings.get("min_edge_percent", 8.0) or 8.0)
            min_conf = float(wf_settings.get("min_confidence", 0.6) or 0.6)
            max_edge_evals_per_cluster = int(wf_settings.get("max_edge_evals_per_article", 3) or 3)
            cache_ttl_minutes = int(wf_settings.get("cache_ttl_minutes", 30) or 30)

            all_findings: list[WorkflowFinding] = []
            market_sources_seen: dict[str, set[str]] = defaultdict(set)
            cycle_deadline = started_at + timedelta(
                seconds=float(wf_settings.get("max_cycle_seconds", _MAX_WORKFLOW_CYCLE_SECONDS) or _MAX_WORKFLOW_CYCLE_SECONDS)
            )
            time_budget_exhausted = False

            for cluster in clusters:
                if datetime.now(timezone.utc) >= cycle_deadline:
                    time_budget_exhausted = True
                    logger.warning(
                        "News workflow cycle hit time budget (%.1fs); ending current cycle early.",
                        float(wf_settings.get("max_cycle_seconds", _MAX_WORKFLOW_CYCLE_SECONDS) or _MAX_WORKFLOW_CYCLE_SECONDS),
                    )
                    break
                article = cluster.representative
                allow_llm = False
                if event_llm_used < event_llm_quota and budget.reserve_calls(1) == 1:
                    allow_llm = True
                    event_llm_used += 1
                event = await event_extractor.extract(
                    title=cluster.headline or article.title,
                    summary=(cluster.summary or cluster.merged_text or article.summary or "")[:1000],
                    source=cluster.primary_source or article.source,
                    model=wf_settings.get("model"),
                    allow_llm=allow_llm,
                )
                if event.confidence < 0.2:
                    continue
                cluster_events += 1

                article_text = (
                    cluster.merged_text
                    or f"{cluster.headline} {cluster.summary or ''}".strip()
                    or f"{article.title} {article.summary or ''}".strip()
                )
                candidates = retriever.retrieve(
                    event=event,
                    article_text=article_text,
                    top_k=top_k,
                    keyword_weight=kw_weight,
                    semantic_weight=sem_weight,
                    event_weight=evt_weight,
                    min_liquidity=market_min_liquidity,
                    similarity_threshold=sim_threshold,
                    min_keyword_signal=min_keyword_signal,
                    min_semantic_signal=min_semantic_signal,
                    min_text_overlap_tokens=1,
                )
                if not candidates:
                    continue

                use_llm_rerank = self._should_use_llm_rerank(candidates)
                allow_llm_rerank = False
                if use_llm_rerank and rerank_llm_used < rerank_llm_quota and budget.reserve_calls(1) == 1:
                    allow_llm_rerank = True
                    rerank_llm_used += 1
                reranked = await reranker.rerank(
                    article_title=cluster.headline or article.title,
                    article_summary=(cluster.summary or cluster.merged_text or article.summary or "")[:900],
                    candidates=candidates,
                    top_n=rerank_top_n,
                    model=wf_settings.get("model"),
                    allow_llm=allow_llm_rerank,
                )
                if not reranked:
                    continue

                reranked = [r for r in reranked if r.rerank_score >= max(0.2, sim_threshold * 0.7)]
                if not reranked:
                    continue

                if bool(wf_settings.get("require_verifier", True)):
                    verified, rejected = self._split_verified_candidates(
                        article=article,
                        event=event,
                        reranked=reranked,
                    )
                    for rejected_finding in rejected:
                        self._attach_cluster_metadata(rejected_finding, cluster)
                        self._attach_market_metadata(
                            rejected_finding,
                            market_metadata_by_id.get(rejected_finding.market_id),
                        )
                        self._assign_finding_keys(rejected_finding)
                        all_findings.append(rejected_finding)
                    reranked = verified
                    if not reranked:
                        continue

                aligned_reranked = []
                for rc in reranked:
                    if not self._is_temporally_compatible(article, event, rc.candidate):
                        rejected = self._build_rejected_finding(
                            article=article,
                            event=event,
                            rc=rc,
                            reason="temporal_mismatch",
                        )
                        self._attach_cluster_metadata(rejected, cluster)
                        self._attach_market_metadata(
                            rejected,
                            market_metadata_by_id.get(rejected.market_id),
                        )
                        self._assign_finding_keys(rejected)
                        all_findings.append(rejected)
                        alignment_dropped += 1
                        continue

                    if self._has_event_market_alignment(event, rc.candidate):
                        aligned_reranked.append(rc)
                        continue

                    # Allow very-strong semantic/LLM evidence even without token overlap.
                    if rc.candidate.semantic_score >= 0.35 and rc.relevance >= 0.7:
                        aligned_reranked.append(rc)
                    else:
                        rejected = self._build_rejected_finding(
                            article=article,
                            event=event,
                            rc=rc,
                            reason="entity_alignment_mismatch",
                        )
                        self._attach_cluster_metadata(rejected, cluster)
                        self._attach_market_metadata(
                            rejected,
                            market_metadata_by_id.get(rejected.market_id),
                        )
                        self._assign_finding_keys(rejected)
                        all_findings.append(rejected)
                        alignment_dropped += 1
                if not aligned_reranked:
                    continue
                reranked = aligned_reranked[: max(1, max_edge_evals_per_cluster)]

                # Source-diversity gate for expensive per-market edge calls.
                diversity_gated = []
                for rc in reranked:
                    seen = market_sources_seen.get(rc.market_id, set())
                    cluster_sources = set(cluster.source_keys or [])
                    if not cluster_sources and article.source:
                        cluster_sources = {(article.source or "").strip().lower()}
                    if cluster_sources and cluster_sources.issubset(seen):
                        budget.llm_calls_skipped += 1
                        continue
                    diversity_gated.append(rc)
                if not diversity_gated:
                    continue

                # Reuse recent cached findings (article+market+price bucket) before LLM.
                cache_keys = [
                    self._cache_key(cluster.article_key, rc.market_id, rc.candidate.yes_price) for rc in diversity_gated
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
                        cluster.article_key,
                        rc.market_id,
                        rc.candidate.yes_price,
                    )
                    row = cached.get(cache_key)
                    if row is not None:
                        hit = self._row_to_finding(row)
                        self._attach_cluster_metadata(hit, cluster)
                        cached_hits.append(hit)
                    else:
                        to_estimate.append(rc)

                llm_calls_for_edges = budget.reserve_calls(len(to_estimate))
                findings = await edge_estimator.estimate_batch(
                    article_title=cluster.headline or article.title,
                    article_summary=(cluster.summary or cluster.merged_text or article.summary or "")[:1000],
                    article_source=cluster.primary_source or article.source,
                    article_url=cluster.primary_url or article.url,
                    article_id=cluster.article_key,
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
                    self._attach_cluster_metadata(finding, cluster)
                    self._attach_market_metadata(
                        finding,
                        market_metadata_by_id.get(finding.market_id),
                    )
                    self._assign_finding_keys(finding)
                    market_sources_seen[finding.market_id].update(self._finding_sources(finding))
                all_findings.extend(article_findings)

            deduped_findings = self._dedupe_findings(all_findings)

            if bool(wf_settings.get("require_second_source", False)):
                by_market_sources: dict[str, set[str]] = defaultdict(set)
                for f in deduped_findings:
                    by_market_sources[f.market_id].update(self._finding_sources(f))
                for f in deduped_findings:
                    if len(by_market_sources.get(f.market_id, set())) < 2:
                        f.actionable = False
                        self._mark_finding_rejected(
                            f,
                            reason="insufficient_source_diversity",
                            details="Need at least two independent sources for actionable signal.",
                        )

            # Actionable findings must be backed by multiple corroborating articles.
            for finding in deduped_findings:
                if not finding.actionable:
                    continue
                article_count, source_count = self._supporting_evidence_counts(finding)
                if article_count < 2:
                    finding.actionable = False
                    self._mark_finding_rejected(
                        finding,
                        reason="insufficient_article_support",
                        details=(
                            "Need at least two corroborating articles in the supporting cluster "
                            f"(got {article_count})."
                        ),
                    )
                    continue
                if source_count < 2:
                    finding.actionable = False
                    self._mark_finding_rejected(
                        finding,
                        reason="insufficient_source_diversity",
                        details=(
                            "Need at least two distinct outlets in supporting evidence "
                            f"(got {source_count})."
                        ),
                    )

            actionable = [f for f in deduped_findings if f.actionable]
            intents: list[dict] = []
            if bool(wf_settings.get("orchestrator_enabled", True)):
                intents = await intent_generator.generate(
                    actionable,
                    min_edge=float(wf_settings.get("orchestrator_min_edge", 10.0) or 10.0),
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
                "clusters": len(clusters),
                "events": cluster_events,
                "market_count": len(market_infos),
                "findings": len(deduped_findings),
                "actionable": len(actionable),
                "intents": len(intents),
                "llm_calls_used": budget.llm_calls_used,
                "llm_calls_skipped": budget.llm_calls_skipped,
                "event_llm_quota": event_llm_quota,
                "event_llm_used": event_llm_used,
                "rerank_llm_quota": rerank_llm_quota,
                "rerank_llm_used": rerank_llm_used,
                "local_model_mode": local_model_mode,
                "cluster_limit": cluster_limit,
                "alignment_dropped": alignment_dropped,
                "time_budget_exhausted": time_budget_exhausted,
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
                "News workflow cycle #%d: %d articles -> %d clusters -> %d findings (%d actionable) -> %d intents (%.1fs)",
                self._cycle_count,
                len(articles),
                len(clusters),
                len(deduped_findings),
                len(actionable),
                len(intents),
                elapsed,
            )

            return {
                "status": "completed",
                "articles": len(articles),
                "clusters": len(clusters),
                "events": cluster_events,
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

    async def _build_market_infos(
        self,
        session: AsyncSession,
        min_liquidity: float = 500.0,
        max_days_to_resolution: int = 365,
    ) -> list[dict]:
        """Build market info from live markets with scanner fallback."""
        try:
            infos = await self._build_market_infos_from_polymarket(
                min_liquidity=min_liquidity,
                max_days_to_resolution=max_days_to_resolution,
            )
            if infos:
                return infos
        except Exception as exc:
            logger.warning("Live market universe build failed: %s", exc)
        logger.warning("Falling back to scanner snapshot for market universe")
        return await self._build_market_infos_from_scanner(
            session,
            min_liquidity=min_liquidity,
            max_days_to_resolution=max_days_to_resolution,
        )

    async def _build_market_infos_from_polymarket(
        self,
        min_liquidity: float,
        max_days_to_resolution: int,
    ) -> list[dict]:
        from services.polymarket import polymarket_client

        markets_task = polymarket_client.get_all_markets(active=True)
        events_task = polymarket_client.get_all_events(closed=False)
        markets, events = await asyncio.gather(markets_task, events_task)

        event_by_slug: dict[str, dict] = {}
        market_to_event: dict[str, dict] = {}
        for event in events or []:
            slug = str(getattr(event, "slug", "") or "")
            title = str(getattr(event, "title", "") or "")
            category = str(getattr(event, "category", "") or "")
            tags = self._normalize_tags(getattr(event, "tags", []))
            meta = {
                "event_slug": slug,
                "event_title": title,
                "category": category,
                "tags": tags,
            }
            if slug:
                event_by_slug[slug] = meta
            for ev_market in getattr(event, "markets", []) or []:
                for key in [
                    str(getattr(ev_market, "id", "") or ""),
                    str(getattr(ev_market, "condition_id", "") or ""),
                ]:
                    if key:
                        market_to_event[key] = meta

        infos: list[dict] = []
        seen: set[str] = set()
        now = datetime.now(timezone.utc)

        for market in markets or []:
            if not bool(getattr(market, "active", True)):
                continue

            question = str(getattr(market, "question", "") or "").strip()
            if len(question) < 20:
                continue

            liquidity = float(getattr(market, "liquidity", 0.0) or 0.0)
            if liquidity < min_liquidity:
                continue

            market_end = self._coerce_datetime(getattr(market, "end_date", None))
            if market_end is not None:
                if market_end < now:
                    continue
                if max_days_to_resolution > 0:
                    if (market_end - now) > timedelta(days=max_days_to_resolution):
                        continue

            market_id = str(getattr(market, "id", "") or "").strip()
            if not market_id:
                market_id = str(getattr(market, "condition_id", "") or "").strip()
            if not market_id or market_id in seen:
                continue
            seen.add(market_id)

            event_slug = str(getattr(market, "event_slug", "") or "").strip()
            event_meta = (
                event_by_slug.get(event_slug)
                or market_to_event.get(market_id)
                or market_to_event.get(str(getattr(market, "condition_id", "") or ""))
                or {}
            )
            tags = self._normalize_tags(
                list(getattr(market, "tags", []) or []) + list(event_meta.get("tags", []) or [])
            )
            token_ids = [str(t) for t in list(getattr(market, "clob_token_ids", []) or []) if t]

            infos.append(
                {
                    "market_id": market_id,
                    "question": question,
                    "event_title": str(event_meta.get("event_title") or ""),
                    "event_slug": event_slug or event_meta.get("event_slug"),
                    "event_ticker": None,
                    "platform": "polymarket",
                    "category": str(event_meta.get("category") or ""),
                    "yes_price": float(getattr(market, "yes_price", 0.5) or 0.5),
                    "no_price": float(getattr(market, "no_price", 0.5) or 0.5),
                    "liquidity": liquidity,
                    "slug": str(getattr(market, "slug", "") or ""),
                    "token_ids": token_ids,
                    "end_date": market_end.isoformat() if market_end else None,
                    "tags": tags,
                }
            )

        return infos

    async def _build_market_infos_from_scanner(
        self,
        session: AsyncSession,
        min_liquidity: float,
        max_days_to_resolution: int,
    ) -> list[dict]:
        """Fallback market universe from scanner DB snapshot opportunities."""
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

                liquidity = float(m.get("liquidity", 0.0) or 0.0)
                if liquidity < min_liquidity:
                    continue

                question = str(m.get("question") or "").strip()
                if len(question) < 20:
                    continue

                market_end = self._coerce_datetime(m.get("end_date"))
                if market_end is not None:
                    now = datetime.now(timezone.utc)
                    if market_end < now:
                        continue
                    if max_days_to_resolution > 0 and (market_end - now) > timedelta(days=max_days_to_resolution):
                        continue

                seen.add(market_id)
                infos.append(
                    {
                        "market_id": market_id,
                        "question": question,
                        "event_title": str(getattr(opp, "event_title", "") or ""),
                        "event_slug": event_slug,
                        "event_ticker": m.get("event_ticker") or m.get("eventTicker"),
                        "platform": str(m.get("platform") or "polymarket"),
                        "category": str(getattr(opp, "category", "") or ""),
                        "yes_price": float(m.get("yes_price", 0.5) or 0.5),
                        "no_price": float(m.get("no_price", 0.5) or 0.5),
                        "liquidity": liquidity,
                        "slug": m.get("slug"),
                        "token_ids": token_ids,
                        "end_date": market_end.isoformat() if market_end else None,
                        "tags": self._normalize_tags(m.get("tags") or []),
                    }
                )

        return infos

    @staticmethod
    def _normalize_tags(raw_tags) -> list[str]:
        out: list[str] = []
        if isinstance(raw_tags, (str, bytes)):
            raw_tags = [raw_tags]
        for tag in raw_tags or []:
            value = ""
            if isinstance(tag, str):
                value = tag
            elif isinstance(tag, dict):
                value = str(tag.get("label") or tag.get("name") or "").strip()
            if not value:
                continue
            if value not in out:
                out.append(value)
        return out

    @staticmethod
    def _coerce_datetime(value: object) -> Optional[datetime]:
        if value is None:
            return None
        if isinstance(value, datetime):
            dt = value
        elif isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            try:
                dt = datetime.fromisoformat(text)
            except ValueError:
                for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
                    try:
                        dt = datetime.strptime(text, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    return None
        else:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt

    @classmethod
    def _is_temporally_compatible(cls, article, event, candidate) -> bool:
        market_end = cls._coerce_datetime(getattr(candidate, "end_date", None))
        if market_end is None:
            return True

        event_date = cls._coerce_datetime(getattr(event, "date", None))
        article_published = cls._coerce_datetime(getattr(article, "published", None))
        article_fetched = cls._coerce_datetime(getattr(article, "fetched_at", None))
        reference_dt = event_date or article_published or article_fetched
        if reference_dt is None:
            return True
        return reference_dt <= (market_end + timedelta(hours=36))

    @staticmethod
    def _build_rejected_finding(article, event, rc, reason: str):
        from services.news.edge_estimator import WorkflowFinding

        c = rc.candidate
        return WorkflowFinding(
            article_id=article.article_id,
            market_id=c.market_id,
            article_title=article.title,
            article_source=article.source,
            article_url=article.url,
            market_question=c.question,
            market_price=float(c.yes_price or 0.5),
            model_probability=float(c.yes_price or 0.5),
            edge_percent=0.0,
            direction="buy_yes",
            confidence=0.0,
            retrieval_score=float(c.combined_score or 0.0),
            semantic_score=float(c.semantic_score or 0.0),
            keyword_score=float(c.keyword_score or 0.0),
            event_score=float(c.event_score or 0.0),
            rerank_score=float(rc.rerank_score or 0.0),
            event_graph={
                "event_type": getattr(event, "event_type", "other"),
                "actors": list(getattr(event, "actors", []) or []),
                "action": getattr(event, "action", ""),
                "date": getattr(event, "date", None),
                "region": getattr(event, "region", None),
                "impact_direction": getattr(event, "impact_direction", None),
                "key_entities": list(getattr(event, "key_entities", []) or []),
            },
            evidence={
                "retrieval": {
                    "keyword_score": round(float(c.keyword_score or 0.0), 4),
                    "semantic_score": round(float(c.semantic_score or 0.0), 4),
                    "event_score": round(float(c.event_score or 0.0), 4),
                    "combined_score": round(float(c.combined_score or 0.0), 4),
                },
                "rerank": {
                    "relevance": round(float(rc.relevance or 0.0), 4),
                    "rationale": rc.rationale,
                    "rerank_score": round(float(rc.rerank_score or 0.0), 4),
                    "used_llm": bool(getattr(rc, "used_llm", False)),
                },
                "rejection_reasons": [reason],
            },
            reasoning=f"Rejected before edge estimation: {reason}.",
            actionable=False,
            created_at=datetime.now(timezone.utc),
        )

    def _split_verified_candidates(self, article, event, reranked):
        verified = []
        rejected = []
        for rc in reranked:
            if getattr(rc, "used_llm", False):
                verified.append(rc)
                continue
            rejected.append(
                self._build_rejected_finding(
                    article=article,
                    event=event,
                    rc=rc,
                    reason="verifier_unavailable",
                )
            )
        return verified, rejected

    @staticmethod
    def _attach_cluster_metadata(finding, cluster) -> None:
        """Attach cluster-level context to a finding in-place."""
        if finding is None or cluster is None:
            return

        finding.article_id = getattr(cluster, "article_key", finding.article_id)
        if getattr(cluster, "headline", None):
            finding.article_title = cluster.headline
        if getattr(cluster, "primary_source", None):
            finding.article_source = cluster.primary_source
        if getattr(cluster, "primary_url", None):
            finding.article_url = cluster.primary_url

        def _iso(dt: Optional[datetime]) -> Optional[str]:
            if dt is None:
                return None
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return dt.replace(tzinfo=None).isoformat() + "Z"

        cluster_meta = {
            "cluster_id": getattr(cluster, "cluster_id", None),
            "article_ids": list(getattr(cluster, "article_ids", []) or []),
            "article_count": int(getattr(cluster, "article_count", 0) or 0),
            "sources": list(getattr(cluster, "source_list", []) or []),
            "article_refs": [
                {
                    "article_id": str(getattr(article, "article_id", "") or ""),
                    "title": str(getattr(article, "title", "") or ""),
                    "url": str(getattr(article, "url", "") or ""),
                    "source": str(getattr(article, "source", "") or ""),
                    "published": _iso(WorkflowOrchestrator._coerce_datetime(getattr(article, "published", None))),
                    "fetched_at": _iso(WorkflowOrchestrator._coerce_datetime(getattr(article, "fetched_at", None))),
                }
                for article in list(getattr(cluster, "articles", []) or [])[:16]
            ],
            "newest_ts": _iso(getattr(cluster, "newest_ts", None)),
            "oldest_ts": _iso(getattr(cluster, "oldest_ts", None)),
        }

        event_graph = dict(getattr(finding, "event_graph", {}) or {})
        event_graph["cluster"] = cluster_meta
        finding.event_graph = event_graph

        evidence = dict(getattr(finding, "evidence", {}) or {})
        evidence["cluster"] = cluster_meta
        finding.evidence = evidence

    @staticmethod
    def _attach_market_metadata(finding, market_metadata: Optional[dict[str, Any]]) -> None:
        """Attach normalized market context to a finding for downstream link rendering."""
        if finding is None or not isinstance(market_metadata, dict):
            return

        market_id = str(
            market_metadata.get("id")
            or market_metadata.get("market_id")
            or getattr(finding, "market_id", "")
            or ""
        ).strip()
        if not market_id:
            return

        market_meta = {
            "id": market_id,
            "slug": str(market_metadata.get("slug") or "").strip() or None,
            "event_slug": str(market_metadata.get("event_slug") or "").strip() or None,
            "event_ticker": str(market_metadata.get("event_ticker") or "").strip() or None,
            "platform": str(market_metadata.get("platform") or "").strip().lower() or None,
            "question": str(
                market_metadata.get("question")
                or getattr(finding, "market_question", "")
                or ""
            ).strip() or None,
        }
        if not any(v for k, v in market_meta.items() if k != "id"):
            return

        evidence = dict(getattr(finding, "evidence", {}) or {})
        existing_market_evidence = evidence.get("market")
        if not isinstance(existing_market_evidence, dict):
            existing_market_evidence = {}
        evidence["market"] = {**existing_market_evidence, **market_meta}
        finding.evidence = evidence

        event_graph = dict(getattr(finding, "event_graph", {}) or {})
        existing_market_graph = event_graph.get("market")
        if not isinstance(existing_market_graph, dict):
            existing_market_graph = {}
        event_graph["market"] = {**existing_market_graph, **market_meta}
        finding.event_graph = event_graph

    @staticmethod
    def _finding_sources(finding) -> set[str]:
        sources: set[str] = set()
        src = str(getattr(finding, "article_source", "") or "").strip().lower()
        if src:
            sources.add(src)

        evidence = getattr(finding, "evidence", {}) or {}
        cluster_meta = evidence.get("cluster") if isinstance(evidence, dict) else None
        if isinstance(cluster_meta, dict):
            for source in cluster_meta.get("sources", []) or []:
                value = str(source or "").strip().lower()
                if value:
                    sources.add(value)
        return sources

    @staticmethod
    def _supporting_evidence_counts(finding) -> tuple[int, int]:
        evidence = getattr(finding, "evidence", {}) or {}
        cluster = evidence.get("cluster") if isinstance(evidence, dict) else None

        article_keys: set[str] = set()
        source_keys: set[str] = set()

        if isinstance(cluster, dict):
            refs = cluster.get("article_refs")
            if isinstance(refs, list):
                for ref in refs:
                    if not isinstance(ref, dict):
                        continue
                    article_key = (
                        str(ref.get("article_id") or "").strip()
                        or str(ref.get("url") or "").strip()
                        or str(ref.get("title") or "").strip().lower()
                    )
                    if article_key:
                        article_keys.add(article_key)
                    source = str(ref.get("source") or "").strip().lower()
                    if source:
                        source_keys.add(source)

            if not article_keys:
                for raw_id in list(cluster.get("article_ids") or []):
                    article_id = str(raw_id or "").strip()
                    if article_id:
                        article_keys.add(article_id)
            if not source_keys:
                for raw_source in list(cluster.get("sources") or []):
                    source = str(raw_source or "").strip().lower()
                    if source:
                        source_keys.add(source)

        if not article_keys:
            fallback_article = (
                str(getattr(finding, "article_id", "") or "").strip()
                or str(getattr(finding, "article_url", "") or "").strip()
                or str(getattr(finding, "article_title", "") or "").strip().lower()
            )
            if fallback_article:
                article_keys.add(fallback_article)

        if not source_keys:
            source = str(getattr(finding, "article_source", "") or "").strip().lower()
            if source:
                source_keys.add(source)

        return len(article_keys), len(source_keys)

    @staticmethod
    def _mark_finding_rejected(finding, reason: str, details: str = "") -> None:
        if finding is None:
            return
        evidence = dict(getattr(finding, "evidence", {}) or {})
        reasons = evidence.get("rejection_reasons")
        reason_list = list(reasons) if isinstance(reasons, list) else []
        if reason not in reason_list:
            reason_list.append(reason)
        evidence["rejection_reasons"] = reason_list
        finding.evidence = evidence

        details = details.strip()
        if not details:
            details = f"Rejected: {reason}."
        current_reasoning = str(getattr(finding, "reasoning", "") or "").strip()
        if current_reasoning:
            if details not in current_reasoning:
                finding.reasoning = f"{current_reasoning} {details}".strip()
        else:
            finding.reasoning = details

    @staticmethod
    def _is_local_model_mode(model: Optional[str], usage: dict[str, Any]) -> bool:
        model_name = str(model or usage.get("active_model") or "").strip().lower()
        configured = {
            str(provider).strip().lower() for provider in list(usage.get("configured_providers", []) or []) if provider
        }

        if model_name.startswith("ollama/") or model_name.startswith("lmstudio/"):
            return True
        if configured and configured.issubset({"ollama", "lmstudio"}):
            return True
        if configured.intersection({"ollama", "lmstudio"}) and any(
            hint in model_name
            for hint in (
                "llama",
                "mistral",
                "qwen",
                "phi",
                "gemma",
                "local",
            )
        ):
            return True
        return False

    async def _hourly_news_spend_usd(self, session: AsyncSession) -> float:
        cutoff = utcnow() - timedelta(hours=1)
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
    def _has_event_market_alignment(event, candidate) -> bool:
        """Require at least one entity/action token overlap with market text."""
        from services.news.market_watcher_index import _tokenize

        generic_tokens = {
            "news",
            "media",
            "daily",
            "weekly",
            "public",
            "times",
            "post",
            "press",
            "radio",
            "television",
            "report",
            "reports",
        }

        def _source_like(value: str) -> bool:
            text = value.strip().lower()
            if not text:
                return True
            if "." in text and " " not in text:
                return True
            return bool(re.search(r"\b(news|media|times|post|observer|press|radio|tv|herald)\b", text))

        event_terms = []
        for value in list(getattr(event, "key_entities", []) or []) + list(getattr(event, "actors", []) or []):
            if isinstance(value, str) and value.strip():
                if _source_like(value):
                    continue
                event_terms.append(value)
        action = getattr(event, "action", "")
        if isinstance(action, str) and action.strip():
            event_terms.append(action)

        if not event_terms:
            return False

        event_tokens = {tok for tok in _tokenize(" ".join(event_terms)) if tok not in generic_tokens}
        if not event_tokens:
            return False

        market_tokens = set(
            _tokenize(
                " ".join(
                    [
                        str(getattr(candidate, "question", "") or ""),
                        str(getattr(candidate, "event_title", "") or ""),
                        str(getattr(candidate, "slug", "") or ""),
                        " ".join(list(getattr(candidate, "tags", []) or [])),
                    ]
                )
            )
        )
        if not market_tokens:
            return False
        return len(event_tokens.intersection(market_tokens)) > 0

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
            .where(NewsWorkflowFinding.confidence > 0.0)
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
            stmt = (
                sqlite_insert(NewsWorkflowFinding)
                .values(
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
                )
                .on_conflict_do_update(
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
            existing_result = await session.execute(query)
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
