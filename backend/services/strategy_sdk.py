"""
Strategy SDK — high-level helpers for custom strategy development.

This module provides a clean, stable API surface for strategy authors.
Import it in your strategy code with:

    from services.strategy_sdk import StrategySDK

All methods are designed to be safe and handle missing data gracefully.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class StrategySDK:
    """High-level helpers for strategy authors.

    All methods are static or class methods — no instantiation needed.

    Usage in opportunity strategies (detect method):
        from services.strategy_sdk import StrategySDK

        price = StrategySDK.get_live_price(market, prices)
        spread = StrategySDK.get_spread_bps(market, prices)

    Usage for LLM calls (async):
        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            StrategySDK.ask_llm("Analyze this market situation...")
        )
    """

    # ── Price helpers ──────────────────────────────────────────────

    @staticmethod
    def get_live_price(market: Any, prices: dict[str, dict], side: str = "YES") -> float:
        """Get the best available mid price for a market.

        Uses live CLOB prices when available, falls back to API prices.

        Args:
            market: A Market object with clob_token_ids and outcome_prices.
            prices: The prices dict passed to detect().
            side: 'YES' or 'NO'.

        Returns:
            The mid price as a float (0.0 if unavailable).
        """
        idx = 0 if side.upper() == "YES" else 1
        fallback = 0.0
        if hasattr(market, "outcome_prices") and len(market.outcome_prices) > idx:
            fallback = market.outcome_prices[idx]

        token_ids = getattr(market, "clob_token_ids", None) or []
        if len(token_ids) > idx:
            token_id = token_ids[idx]
            if token_id in prices:
                return prices[token_id].get("mid", fallback)
        return fallback

    @staticmethod
    def get_spread_bps(market: Any, prices: dict[str, dict], side: str = "YES") -> Optional[float]:
        """Get the bid-ask spread in basis points for a market side.

        Args:
            market: A Market object.
            prices: The prices dict passed to detect().
            side: 'YES' or 'NO'.

        Returns:
            Spread in basis points, or None if data unavailable.
        """
        idx = 0 if side.upper() == "YES" else 1
        token_ids = getattr(market, "clob_token_ids", None) or []
        if len(token_ids) <= idx:
            return None
        token_id = token_ids[idx]
        data = prices.get(token_id)
        if not data:
            return None
        bid = data.get("best_bid", 0)
        ask = data.get("best_ask", 0)
        mid = data.get("mid", 0)
        if mid <= 0 or bid <= 0 or ask <= 0:
            return None
        return ((ask - bid) / mid) * 10000

    @staticmethod
    def get_ws_mid_price(token_id: str) -> Optional[float]:
        """Get the real-time WebSocket mid price for a token.

        Args:
            token_id: The CLOB token ID.

        Returns:
            Mid price or None if not fresh.
        """
        try:
            from services.ws_feeds import get_feed_manager

            fm = get_feed_manager()
            if fm and fm.cache.is_fresh(token_id):
                return fm.cache.get_mid_price(token_id)
        except Exception:
            pass
        return None

    @staticmethod
    def get_ws_spread_bps(token_id: str) -> Optional[float]:
        """Get the real-time WebSocket spread in basis points.

        Args:
            token_id: The CLOB token ID.

        Returns:
            Spread in bps or None.
        """
        try:
            from services.ws_feeds import get_feed_manager

            fm = get_feed_manager()
            if fm:
                return fm.cache.get_spread_bps(token_id)
        except Exception:
            pass
        return None

    # ── Chainlink oracle ──────────────────────────────────────────

    @staticmethod
    def get_chainlink_price(asset: str) -> Optional[float]:
        """Get the latest Chainlink oracle price for a crypto asset.

        Args:
            asset: Asset symbol — 'BTC', 'ETH', 'SOL', or 'XRP'.

        Returns:
            The oracle price in USD, or None if unavailable.
        """
        try:
            from services.chainlink_feed import get_chainlink_feed

            feed = get_chainlink_feed()
            if feed:
                p = feed.get_price(asset.upper())
                return p.price if p else None
        except Exception:
            pass
        return None

    # ── LLM helpers ───────────────────────────────────────────────

    @staticmethod
    async def ask_llm(
        prompt: str,
        model: str = "gpt-4o-mini",
        system: str = "",
        purpose: str = "custom_strategy",
    ) -> str:
        """Send a prompt to an LLM and get the text response.

        Args:
            prompt: The user prompt.
            model: Model identifier (e.g. 'gpt-4o-mini', 'claude-haiku-4-5-20251001').
            system: Optional system prompt.
            purpose: Usage attribution tag (default: 'custom_strategy').

        Returns:
            The LLM response text, or empty string on failure.
        """
        try:
            from services.ai import get_llm_manager
            from services.ai.llm_provider import LLMMessage

            manager = get_llm_manager()
            if not manager.is_available():
                return ""

            messages = []
            if system:
                messages.append(LLMMessage(role="system", content=system))
            messages.append(LLMMessage(role="user", content=prompt))

            response = await manager.chat(
                messages=messages,
                model=model,
                purpose=purpose,
            )
            return response.content
        except Exception as e:
            logger.warning("StrategySDK.ask_llm failed: %s", e)
            return ""

    @staticmethod
    async def ask_llm_json(
        prompt: str,
        schema: dict,
        model: str = "gpt-4o-mini",
        system: str = "",
        purpose: str = "custom_strategy",
    ) -> dict:
        """Get structured JSON output from an LLM.

        Args:
            prompt: The user prompt.
            schema: JSON Schema the response must conform to.
            model: Model identifier.
            system: Optional system prompt.
            purpose: Usage attribution tag.

        Returns:
            Parsed dict conforming to the schema, or empty dict on failure.
        """
        try:
            from services.ai import get_llm_manager
            from services.ai.llm_provider import LLMMessage

            manager = get_llm_manager()
            if not manager.is_available():
                return {}

            messages = []
            if system:
                messages.append(LLMMessage(role="system", content=system))
            messages.append(LLMMessage(role="user", content=prompt))

            return await manager.structured_output(
                messages=messages,
                schema=schema,
                model=model,
                purpose=purpose,
            )
        except Exception as e:
            logger.warning("StrategySDK.ask_llm_json failed: %s", e)
            return {}

    # ── Fee calculation ───────────────────────────────────────────

    @staticmethod
    def calculate_fees(
        total_cost: float,
        expected_payout: float = 1.0,
        n_legs: int = 1,
    ) -> Optional[dict]:
        """Calculate comprehensive fees for a potential trade.

        Args:
            total_cost: Total cost to enter the position.
            expected_payout: Expected payout if the trade succeeds.
            n_legs: Number of legs in the trade.

        Returns:
            Dict with fee breakdown, or None on failure.
        """
        try:
            from services.fee_model import fee_model

            breakdown = fee_model.full_breakdown(
                total_cost=total_cost,
                expected_payout=expected_payout,
                n_legs=n_legs,
            )
            return {
                "winner_fee": breakdown.winner_fee,
                "gas_cost_usd": breakdown.gas_cost_usd,
                "spread_cost": breakdown.spread_cost,
                "multi_leg_slippage": breakdown.multi_leg_slippage,
                "total_fees": breakdown.total_fees,
                "fee_as_pct_of_payout": breakdown.fee_as_pct_of_payout,
            }
        except Exception as e:
            logger.warning("StrategySDK.calculate_fees failed: %s", e)
            return None

    # ── Market filtering helpers ──────────────────────────────────

    @staticmethod
    def active_binary_markets(markets: list) -> list:
        """Filter to only active, non-closed, binary (2-outcome) markets.

        Args:
            markets: List of Market objects.

        Returns:
            Filtered list of markets.
        """
        return [
            m
            for m in markets
            if getattr(m, "active", False)
            and not getattr(m, "closed", False)
            and len(getattr(m, "outcome_prices", []) or []) == 2
        ]

    @staticmethod
    def markets_by_category(events: list, category: str) -> list:
        """Get all active markets belonging to events of a given category.

        Args:
            events: List of Event objects.
            category: Category to filter by (e.g. 'crypto', 'politics').

        Returns:
            List of Market objects in matching events.
        """
        result = []
        for event in events:
            if getattr(event, "category", None) == category:
                for market in getattr(event, "markets", []) or []:
                    if getattr(market, "active", False) and not getattr(market, "closed", False):
                        result.append(market)
        return result

    @staticmethod
    def find_event_for_market(market: Any, events: list) -> Optional[Any]:
        """Find the parent Event for a given Market.

        Args:
            market: A Market object.
            events: List of Event objects.

        Returns:
            The parent Event, or None.
        """
        market_id = getattr(market, "id", None)
        if not market_id:
            return None
        for event in events:
            for m in getattr(event, "markets", []) or []:
                if getattr(m, "id", None) == market_id:
                    return event
        return None

    # ── Order book depth ───────────────────────────────────────

    @staticmethod
    def get_order_book_depth(
        market: Any,
        side: str = "YES",
        size_usd: float = 100.0,
    ) -> Optional[dict]:
        """Get order book depth analysis for a market side.

        Uses the internal VWAP calculator to estimate execution cost,
        slippage, and fill probability for a given position size.

        Args:
            market: A Market object.
            side: 'YES' or 'NO'.
            size_usd: Position size in USD to estimate execution for.

        Returns:
            Dict with depth analysis, or None if unavailable:
            {
                "vwap_price": float,       # Volume-weighted avg price
                "slippage_bps": float,     # Slippage in basis points
                "fill_probability": float, # Estimated fill probability (0-1)
                "available_liquidity": float,  # Liquidity at this level
            }
        """
        try:
            from services.optimization.vwap import vwap_calculator

            if not vwap_calculator:
                return None

            token_idx = 0 if side.upper() == "YES" else 1
            token_ids = getattr(market, "clob_token_ids", None) or []
            if len(token_ids) <= token_idx:
                return None

            result = vwap_calculator.estimate_execution(
                token_id=token_ids[token_idx],
                size_usd=size_usd,
            )
            if result is None:
                return None

            return {
                "vwap_price": result.get("vwap_price", 0),
                "slippage_bps": result.get("slippage_bps", 0),
                "fill_probability": result.get("fill_probability", 1.0),
                "available_liquidity": result.get("available_liquidity", 0),
            }
        except Exception:
            return None

    @staticmethod
    def get_book_levels(
        market: Any,
        side: str = "YES",
        max_levels: int = 10,
    ) -> Optional[list[dict]]:
        """Get raw order book levels (bids or asks) for a market side.

        Args:
            market: A Market object.
            side: 'YES' or 'NO'.
            max_levels: Maximum number of price levels to return.

        Returns:
            List of {price, size} dicts sorted by best price, or None.
        """
        try:
            from services.optimization.vwap import vwap_calculator

            if not vwap_calculator:
                return None

            token_idx = 0 if side.upper() == "YES" else 1
            token_ids = getattr(market, "clob_token_ids", None) or []
            if len(token_ids) <= token_idx:
                return None

            return vwap_calculator.get_book_levels(
                token_id=token_ids[token_idx],
                max_levels=max_levels,
            )
        except Exception:
            return None

    # ── Historical price access ────────────────────────────────

    @staticmethod
    def get_price_history(
        token_id: str,
        max_snapshots: int = 60,
    ) -> list[dict]:
        """Get recent price snapshots for a token.

        Returns the most recent snapshots from the WebSocket feed cache.
        Each snapshot includes timestamp, mid, bid, ask.

        Args:
            token_id: The CLOB token ID.
            max_snapshots: Maximum number of snapshots to return.

        Returns:
            List of price snapshots (newest first), each:
            {"timestamp": str, "mid": float, "bid": float, "ask": float}
            Empty list if no history available.
        """
        try:
            from services.ws_feeds import get_feed_manager

            fm = get_feed_manager()
            if fm and hasattr(fm.cache, "get_price_history"):
                return fm.cache.get_price_history(token_id, max_snapshots=max_snapshots)
        except Exception:
            pass
        return []

    @staticmethod
    def get_price_change(
        token_id: str,
        lookback_seconds: int = 300,
    ) -> Optional[dict]:
        """Get price change over a lookback period.

        Args:
            token_id: The CLOB token ID.
            lookback_seconds: How far back to look (default 5 minutes).

        Returns:
            Dict with price change info, or None:
            {
                "current_mid": float,
                "prior_mid": float,
                "change_abs": float,
                "change_pct": float,
                "snapshots_in_window": int,
            }
        """
        try:
            from services.ws_feeds import get_feed_manager

            fm = get_feed_manager()
            if fm and hasattr(fm.cache, "get_price_change"):
                return fm.cache.get_price_change(token_id, lookback_seconds=lookback_seconds)
        except Exception:
            pass
        return None

    # ── News data access ───────────────────────────────────────

    @staticmethod
    def get_recent_news(
        query: str = "",
        max_articles: int = 20,
    ) -> list[dict]:
        """Get recent news articles, optionally filtered by query.

        Args:
            query: Optional search query to filter articles.
            max_articles: Maximum number of articles to return.

        Returns:
            List of article dicts with title, source, published_at, summary.
        """
        try:
            from services.news.feed_service import news_feed_service

            articles = news_feed_service.search_articles(query=query, limit=max_articles)
            return [
                {
                    "title": getattr(a, "title", ""),
                    "source": getattr(a, "source", ""),
                    "published_at": str(getattr(a, "published_at", "")),
                    "summary": getattr(a, "summary", getattr(a, "description", "")),
                    "url": getattr(a, "url", getattr(a, "link", "")),
                }
                for a in (articles or [])
            ]
        except Exception:
            return []

    @staticmethod
    def get_news_for_market(
        market: Any,
        max_articles: int = 10,
    ) -> list[dict]:
        """Get news articles semantically matched to a specific market.

        Args:
            market: A Market object.
            max_articles: Maximum number of articles to return.

        Returns:
            List of article dicts with relevance_score.
        """
        try:
            from services.news.semantic_matcher import semantic_matcher

            question = getattr(market, "question", "")
            if not question:
                return []

            matches = semantic_matcher.find_matches(question, top_k=max_articles)
            return [
                {
                    "title": m.get("title", ""),
                    "source": m.get("source", ""),
                    "relevance_score": m.get("score", 0),
                    "published_at": str(m.get("published_at", "")),
                    "summary": m.get("summary", ""),
                }
                for m in (matches or [])
            ]
        except Exception:
            return []
