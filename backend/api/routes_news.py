"""
API routes for the News Intelligence layer.

Provides endpoints for:
- News feed status and manual fetch
- Semantic matching results
- On-demand forecaster committee analysis
- News-driven edge detection results
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


def _build_market_infos_from_scanner():
    """Build MarketInfo list from the scanner's full cached market universe.

    Uses scanner._cached_markets (populated on every full scan) so that
    news matching works even when no structural arbitrage opportunities
    exist.  Falls back to scanner.get_opportunities() as a last resort.
    """
    from services.news.semantic_matcher import MarketInfo
    from services import scanner as scanner_inst

    # Primary source: the cached market universe from the last full scan.
    # Each item is a models.market.Market instance (Pydantic).
    cached_markets = getattr(scanner_inst, "_cached_markets", [])
    cached_prices = getattr(scanner_inst, "_cached_prices", {})
    cached_events = getattr(scanner_inst, "_cached_events", [])

    if cached_markets:
        # Build event_id -> Event lookup for category / title
        event_by_market: dict[str, object] = {}
        for ev in cached_events:
            for m in ev.markets:
                event_by_market[m.id] = ev

        seen_ids: set[str] = set()
        market_infos = []
        for m in cached_markets:
            if m.id in seen_ids:
                continue
            seen_ids.add(m.id)

            # Derive yes/no prices from outcome_prices or cached live prices
            yes_price = 0.5
            no_price = 0.5
            if m.outcome_prices and len(m.outcome_prices) >= 2:
                yes_price = m.outcome_prices[0]
                no_price = m.outcome_prices[1]
            elif m.clob_token_ids:
                for i, tid in enumerate(m.clob_token_ids):
                    p = cached_prices.get(tid)
                    if p is not None:
                        if i == 0:
                            yes_price = p
                        else:
                            no_price = p

            ev = event_by_market.get(m.id)
            market_infos.append(
                MarketInfo(
                    market_id=m.id,
                    question=m.question,
                    event_title=ev.title if ev else "",
                    category=ev.category or "" if ev else "",
                    yes_price=yes_price,
                    no_price=no_price,
                    liquidity=m.liquidity,
                )
            )
        return market_infos

    # Fallback: extract from current opportunity list
    opps = scanner_inst.get_opportunities()
    seen_ids: set[str] = set()
    market_infos = []
    for opp in opps:
        for m in opp.markets:
            mid = m.get("id", "")
            if mid in seen_ids:
                continue
            seen_ids.add(mid)
            market_infos.append(
                MarketInfo(
                    market_id=mid,
                    question=m.get("question", ""),
                    event_title=opp.event_title or "",
                    category=opp.category or "",
                    yes_price=m.get("yes_price", 0.5),
                    no_price=m.get("no_price", 0.5),
                    liquidity=m.get("liquidity", 0.0),
                )
            )
    return market_infos


# ======================================================================
# News Feed
# ======================================================================


@router.get("/news/feed/status")
async def get_feed_status():
    """Get the current status of the news feed service."""
    from services.news.feed_service import news_feed_service
    from services.news.semantic_matcher import semantic_matcher

    articles = news_feed_service.get_articles()

    # Source breakdown
    sources: dict[str, int] = {}
    for a in articles:
        sources[a.feed_source] = sources.get(a.feed_source, 0) + 1

    return {
        "article_count": len(articles),
        "sources": sources,
        "running": news_feed_service._running,
        "matcher": semantic_matcher.get_status(),
    }


@router.post("/news/feed/fetch")
async def trigger_news_fetch():
    """Manually trigger a news fetch from all sources."""
    from services.news.feed_service import news_feed_service

    try:
        new_articles = await news_feed_service.fetch_all()
        return {
            "new_articles": len(new_articles),
            "total_articles": news_feed_service.article_count,
            "articles": [
                {
                    "title": a.title,
                    "source": a.source,
                    "feed_source": a.feed_source,
                    "url": a.url,
                    "published": a.published.isoformat() if a.published else None,
                    "category": a.category,
                }
                for a in new_articles[:20]  # Return first 20
            ],
        }
    except Exception as e:
        logger.error("News fetch failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/news/feed/articles")
async def get_articles(
    max_age_hours: int = Query(24, ge=1, le=168),
    source: Optional[str] = Query(None, description="Filter by feed source"),
    limit: int = Query(50, ge=1, le=200),
):
    """Get articles currently in the news feed store."""
    from services.news.feed_service import news_feed_service

    articles = news_feed_service.get_articles(max_age_hours=max_age_hours)

    if source:
        articles = [a for a in articles if a.feed_source == source]

    articles.sort(key=lambda a: a.fetched_at.timestamp(), reverse=True)

    return {
        "total": len(articles),
        "articles": [
            {
                "article_id": a.article_id,
                "title": a.title,
                "source": a.source,
                "feed_source": a.feed_source,
                "url": a.url,
                "published": a.published.isoformat() if a.published else None,
                "category": a.category,
                "summary": a.summary[:200] if a.summary else "",
                "has_embedding": a.embedding is not None,
                "fetched_at": a.fetched_at.isoformat(),
            }
            for a in articles[:limit]
        ],
    }


@router.delete("/news/feed/clear")
async def clear_articles():
    """Clear all articles from the news feed store."""
    from services.news.feed_service import news_feed_service

    count = news_feed_service.clear()
    return {"cleared": count}


# ======================================================================
# Semantic Matching
# ======================================================================


class MatchRequest(BaseModel):
    max_age_hours: int = 6
    top_k: int = 3
    threshold: Optional[float] = None


@router.post("/news/match")
async def run_matching(request: MatchRequest):
    """Run semantic matching between current articles and active markets.

    This fetches articles (if needed), embeds them, and matches them
    against the scanner's current markets.
    """
    from services.news.feed_service import news_feed_service
    from services.news.semantic_matcher import semantic_matcher

    try:
        # Fetch if empty
        if news_feed_service.article_count == 0:
            await news_feed_service.fetch_all()

        articles = news_feed_service.get_articles(max_age_hours=request.max_age_hours)
        if not articles:
            return {"matches": [], "message": "No articles available"}

        # Build market index from the full active market universe
        market_infos = _build_market_infos_from_scanner()

        if not market_infos:
            return {"matches": [], "message": "No markets available from scanner"}

        if not semantic_matcher._initialized:
            semantic_matcher.initialize()

        semantic_matcher.update_market_index(market_infos)
        semantic_matcher.embed_articles(articles)

        matches = semantic_matcher.match_articles_to_markets(
            articles, top_k=request.top_k, threshold=request.threshold
        )

        return {
            "total_articles": len(articles),
            "total_markets": len(market_infos),
            "total_matches": len(matches),
            "matcher_mode": "semantic" if semantic_matcher.is_ml_mode else "tfidf",
            "matches": [
                {
                    "article_title": m.article.title,
                    "article_source": m.article.source,
                    "article_url": m.article.url,
                    "market_question": m.market.question,
                    "market_id": m.market.market_id,
                    "market_price": m.market.yes_price,
                    "similarity": round(m.similarity, 4),
                    "match_method": m.match_method,
                }
                for m in matches[:50]
            ],
        }

    except Exception as e:
        logger.error("Matching failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ======================================================================
# Edge Detection
# ======================================================================


class EdgeDetectionRequest(BaseModel):
    max_age_hours: int = 6
    top_k: int = 3
    threshold: Optional[float] = None
    model: Optional[str] = None


@router.post("/news/edges")
async def detect_edges(request: EdgeDetectionRequest):
    """Run the full news edge detection pipeline.

    Fetches news, matches to markets, estimates probabilities via LLM,
    and returns edges where model diverges from market.
    """
    from services.news.feed_service import news_feed_service
    from services.news.semantic_matcher import semantic_matcher
    from services.news.edge_detector import edge_detector

    try:
        # Fetch if empty
        if news_feed_service.article_count == 0:
            await news_feed_service.fetch_all()

        articles = news_feed_service.get_articles(max_age_hours=request.max_age_hours)
        if not articles:
            return {"edges": [], "message": "No articles available"}

        # Build market index from the full active market universe
        market_infos = _build_market_infos_from_scanner()

        if not market_infos:
            return {"edges": [], "message": "No markets available from scanner"}

        if not semantic_matcher._initialized:
            semantic_matcher.initialize()
        semantic_matcher.update_market_index(market_infos)
        semantic_matcher.embed_articles(articles)

        matches = semantic_matcher.match_articles_to_markets(
            articles, top_k=request.top_k, threshold=request.threshold
        )

        if not matches:
            return {"edges": [], "message": "No matches found"}

        edges = await edge_detector.detect_edges(matches, model=request.model)

        return {
            "total_articles": len(articles),
            "total_markets": len(market_infos),
            "total_matches": len(matches),
            "total_edges": len(edges),
            "edges": [
                {
                    "article_title": e.match.article.title,
                    "article_source": e.match.article.source,
                    "article_url": e.match.article.url,
                    "market_question": e.match.market.question,
                    "market_id": e.match.market.market_id,
                    "market_price": e.market_price,
                    "model_probability": e.model_probability,
                    "edge_percent": round(e.edge_percent, 2),
                    "direction": e.direction,
                    "confidence": round(e.confidence, 2),
                    "reasoning": e.reasoning,
                    "similarity": round(e.match.similarity, 4),
                    "estimated_at": e.estimated_at.isoformat(),
                }
                for e in edges
            ],
        }

    except Exception as e:
        logger.error("Edge detection failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ======================================================================
# Forecaster Committee (On-Demand Deep Analysis)
# ======================================================================


class CommitteeRequest(BaseModel):
    market_question: str
    market_price: float  # Current YES price as decimal (e.g. 0.55)
    news_context: str = ""  # Optional news text to include
    event_title: str = ""
    category: str = ""
    model: Optional[str] = None


@router.post("/news/forecast")
async def run_forecaster_committee(request: CommitteeRequest):
    """Run the multi-agent forecaster committee on a market.

    Deploys three specialized agents (Outside View, Inside View,
    Adversarial Critic) and aggregates their estimates.

    This is the deep-analysis mode — expensive but high quality.
    """
    from services.news.forecaster_committee import forecaster_committee

    try:
        if not (0.01 <= request.market_price <= 0.99):
            raise HTTPException(
                status_code=400,
                detail="market_price must be between 0.01 and 0.99",
            )

        result = await forecaster_committee.analyze(
            market_question=request.market_question,
            market_price=request.market_price,
            news_context=request.news_context,
            event_title=request.event_title,
            category=request.category,
            model=request.model,
        )

        return result.to_dict()

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Forecaster committee failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


class CommitteeMarketRequest(BaseModel):
    """Analyze a market by ID — fetches context automatically."""

    market_id: str
    model: Optional[str] = None
    include_news: bool = True
    max_articles: int = 5


@router.post("/news/forecast/market")
async def forecast_market_by_id(request: CommitteeMarketRequest):
    """Run forecaster committee on a market found in the scanner.

    Automatically gathers market context and recent news.
    """
    from services import scanner
    from services.news.feed_service import news_feed_service
    from services.news.semantic_matcher import semantic_matcher, MarketInfo
    from services.news.forecaster_committee import forecaster_committee

    try:
        # Find the market in scanner opportunities
        opps = scanner.get_opportunities()
        target_opp = None
        target_market = None

        for opp in opps:
            for m in opp.markets:
                if m.get("id") == request.market_id:
                    target_opp = opp
                    target_market = m
                    break
            if target_market:
                break

        if not target_market:
            raise HTTPException(
                status_code=404, detail="Market not found in current scanner results"
            )

        market_question = target_market.get("question", "")
        market_price = target_market.get("yes_price", 0.5)
        event_title = target_opp.event_title or ""
        category = target_opp.category or ""

        # Optionally gather related news
        news_context = ""
        if request.include_news:
            try:
                if news_feed_service.article_count == 0:
                    await news_feed_service.fetch_all()

                articles = news_feed_service.get_articles(max_age_hours=24)

                if articles and semantic_matcher._initialized:
                    mi = MarketInfo(
                        market_id=request.market_id,
                        question=market_question,
                        event_title=event_title,
                        category=category,
                        yes_price=market_price,
                    )
                    semantic_matcher.update_market_index([mi])
                    semantic_matcher.embed_articles(articles)
                    matches = semantic_matcher.match_articles_to_markets(
                        articles, top_k=request.max_articles
                    )

                    if matches:
                        parts = []
                        for i, match in enumerate(matches[: request.max_articles], 1):
                            parts.append(
                                f"Article {i}: {match.article.title} "
                                f"(Source: {match.article.source}, "
                                f"Relevance: {match.similarity:.2f})"
                            )
                            if match.article.summary:
                                parts.append(
                                    f"  Summary: {match.article.summary[:200]}"
                                )
                        news_context = "\n".join(parts)
            except Exception as e:
                logger.debug("News gathering for committee failed: %s", e)

        result = await forecaster_committee.analyze(
            market_question=market_question,
            market_price=market_price,
            news_context=news_context,
            event_title=event_title,
            category=category,
            model=request.model,
        )

        return result.to_dict()

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Market forecast failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
