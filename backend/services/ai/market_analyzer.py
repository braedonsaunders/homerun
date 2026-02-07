"""
Market Analyzer with Sub-Agent Routing.

A mini-agent that receives natural language market questions and
routes them to the appropriate combination of analysis tools:
- Resolution criteria analysis
- News/sentiment search
- Order book/liquidity check
- Market correlation detection
- Historical price analysis

Inspired by virattt/dexter's financial_search sub-agent pattern
where an inner LLM call routes to specialized sub-tools.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Optional

import httpx

from services.ai import get_llm_manager
from services.ai.llm_provider import LLMMessage
from services.ai.scratchpad import ScratchpadService

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GAMMA_API_URL = "https://gamma-api.polymarket.com"
CLOB_API_URL = "https://clob.polymarket.com"

_HTTP_TIMEOUT = 15  # seconds


# ---------------------------------------------------------------------------
# Lazy imports for sibling modules that may not exist yet
# ---------------------------------------------------------------------------


def _get_resolution_analyzer():
    """Lazy-import the resolution analyzer singleton."""
    from services.ai.resolution_analyzer import resolution_analyzer

    return resolution_analyzer


def _get_news_sentiment_analyzer():
    """Lazy-import the news sentiment analyzer singleton."""
    from services.ai.news_sentiment import news_sentiment_analyzer

    return news_sentiment_analyzer


def _get_agent_helpers():
    """Lazy-import Agent framework types and runner."""
    from services.ai.agent import AgentTool, run_agent_to_completion

    return AgentTool, run_agent_to_completion


# ---------------------------------------------------------------------------
# MarketAnalyzer
# ---------------------------------------------------------------------------


class MarketAnalyzer:
    """
    Intelligent market analysis router.

    Routes natural language queries to specialized analysis tools
    and aggregates results.

    Usage:
        analyzer = MarketAnalyzer()
        result = await analyzer.analyze(
            query="Should I bet on this Bitcoin market?",
            market_id="0x...",
            market_question="Will Bitcoin exceed $100K by March 2025?"
        )
    """

    def __init__(self) -> None:
        self._scratchpad = ScratchpadService()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def analyze(
        self,
        query: str,
        market_id: Optional[str] = None,
        market_question: Optional[str] = None,
        context: Optional[dict] = None,
        model: Optional[str] = None,
    ) -> dict:
        """
        Run a full market analysis by routing to relevant sub-tools.

        Uses the Agent with market-specific tools to autonomously
        gather and synthesize information.

        Args:
            query: Natural language question about the market.
            market_id: Optional Polymarket condition ID.
            market_question: Optional market question text for context.
            context: Optional extra context dict (prices, volume, etc.).
            model: LLM model to use. Defaults to ``"gpt-4o-mini"``.

        Returns:
            Dict with keys:
                query        -- the original query
                market_id    -- the market ID (if provided)
                analysis     -- full natural language analysis
                data_gathered-- raw data from tools
                tools_used   -- list of tool names called
                session_id   -- scratchpad session ID
        """
        session_id = uuid.uuid4().hex[:16]

        try:
            AgentTool, run_agent_to_completion = _get_agent_helpers()

            # Build tools list
            tools = self._build_tools(market_id)

            # Build system prompt
            system_prompt = self._build_system_prompt(market_question, context)

            # Run the agent
            result = await run_agent_to_completion(
                system_prompt=system_prompt,
                query=query,
                tools=tools,
                model=model or "gpt-4o-mini",
                max_iterations=8,
                session_type="market_analysis",
                market_id=market_id,
            )

            return result

        except ImportError:
            # Agent framework not available yet -- fall back to a simpler
            # single-shot LLM call with manually gathered data.
            logger.warning(
                "Agent framework not available, falling back to single-shot analysis"
            )
            return await self._fallback_analyze(
                query=query,
                market_id=market_id,
                market_question=market_question,
                context=context,
                model=model,
                session_id=session_id,
            )

        except Exception as exc:
            logger.error("Market analysis failed: %s", exc, exc_info=True)
            return {
                "query": query,
                "market_id": market_id,
                "analysis": f"Analysis failed: {exc}",
                "data_gathered": {},
                "tools_used": [],
                "session_id": session_id,
                "error": str(exc),
            }

    # ------------------------------------------------------------------
    # Tool builder
    # ------------------------------------------------------------------

    def _build_tools(self, market_id: Optional[str] = None) -> list:
        """Build the set of tools available to the market analyzer agent."""
        AgentTool, _ = _get_agent_helpers()
        tools: list = []

        # Tool 1: Fetch market details from Polymarket
        tools.append(
            AgentTool(
                name="get_market_details",
                description=(
                    "Fetch detailed information about a Polymarket market "
                    "including question, description, resolution rules, current "
                    "prices, volume, and liquidity."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "market_id": {
                            "type": "string",
                            "description": "The market's condition ID",
                        },
                    },
                    "required": ["market_id"],
                },
                handler=self._tool_get_market_details,
                max_calls=3,
            )
        )

        # Tool 2: Search for related news
        tools.append(
            AgentTool(
                name="search_news",
                description=(
                    "Search for recent news articles related to a topic or "
                    "market question. Returns headlines, summaries, and "
                    "sentiment analysis."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query for news",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum results to return",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
                handler=self._tool_search_news,
                max_calls=3,
            )
        )

        # Tool 3: Analyze resolution criteria
        tools.append(
            AgentTool(
                name="analyze_resolution",
                description=(
                    "Deep analysis of a market's resolution criteria, checking "
                    "for ambiguities, edge cases, and risks."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "market_id": {"type": "string"},
                        "question": {"type": "string"},
                        "description": {"type": "string", "default": ""},
                        "resolution_source": {"type": "string", "default": ""},
                    },
                    "required": ["market_id", "question"],
                },
                handler=self._tool_analyze_resolution,
                max_calls=2,
            )
        )

        # Tool 4: Check order book depth
        tools.append(
            AgentTool(
                name="check_orderbook",
                description=(
                    "Check the order book depth and liquidity for a market's "
                    "tokens. Returns bid/ask spreads and available liquidity "
                    "at different price levels."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "token_id": {
                            "type": "string",
                            "description": "The token ID to check",
                        },
                    },
                    "required": ["token_id"],
                },
                handler=self._tool_check_orderbook,
                max_calls=4,
            )
        )

        # Tool 5: Find related markets
        tools.append(
            AgentTool(
                name="find_related_markets",
                description=(
                    "Find other Polymarket markets related to the same event "
                    "or topic. Useful for cross-market analysis and correlation "
                    "detection."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "event_id": {
                            "type": "string",
                            "description": "Event ID to find related markets",
                        },
                        "search_query": {
                            "type": "string",
                            "description": "Text search for related markets",
                        },
                    },
                },
                handler=self._tool_find_related_markets,
                max_calls=2,
            )
        )

        return tools

    # ==================================================================
    # Tool Implementations
    # ==================================================================

    async def _tool_get_market_details(self, args: dict) -> dict:
        """Fetch market details from the Polymarket Gamma API."""
        try:
            market_id = args["market_id"]
            async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
                resp = await client.get(
                    f"{GAMMA_API_URL}/markets/{market_id}",
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return {
                        "question": data.get("question", ""),
                        "description": data.get("description", ""),
                        "resolution_source": data.get("resolutionSource", ""),
                        "end_date": data.get("endDate", ""),
                        "outcomes": data.get("outcomes", []),
                        "outcome_prices": data.get("outcomePrices", []),
                        "volume": data.get("volume", 0),
                        "liquidity": data.get("liquidity", 0),
                        "active": data.get("active", False),
                        "closed": data.get("closed", False),
                        "condition_id": data.get("conditionId", ""),
                        "slug": data.get("slug", ""),
                    }

                # Market ID might be a condition_id; search by that instead
                if resp.status_code == 404:
                    resp2 = await client.get(
                        f"{GAMMA_API_URL}/markets",
                        params={"condition_id": market_id, "limit": 1},
                    )
                    if resp2.status_code == 200:
                        results = resp2.json()
                        if results:
                            data = results[0]
                            return {
                                "question": data.get("question", ""),
                                "description": data.get("description", ""),
                                "resolution_source": data.get("resolutionSource", ""),
                                "end_date": data.get("endDate", ""),
                                "outcomes": data.get("outcomes", []),
                                "outcome_prices": data.get("outcomePrices", []),
                                "volume": data.get("volume", 0),
                                "liquidity": data.get("liquidity", 0),
                                "active": data.get("active", False),
                                "closed": data.get("closed", False),
                                "condition_id": data.get("conditionId", ""),
                                "slug": data.get("slug", ""),
                            }

                return {"error": f"Market not found (HTTP {resp.status_code})"}

        except Exception as exc:
            logger.error("get_market_details failed: %s", exc)
            return {"error": str(exc)}

    async def _tool_search_news(self, args: dict) -> dict:
        """Search for news using the NewsSentimentAnalyzer."""
        try:
            query = args["query"]
            max_results = args.get("max_results", 5)

            analyzer = _get_news_sentiment_analyzer()
            result = await analyzer.search_and_analyze(
                query=query,
                max_articles=max_results,
            )
            return result

        except ImportError:
            # news_sentiment module not available -- do a basic RSS fetch
            logger.warning("NewsSentimentAnalyzer not available, using basic RSS fetch")
            return await self._basic_news_search(
                args.get("query", ""), args.get("max_results", 5)
            )

        except Exception as exc:
            logger.error("search_news failed: %s", exc)
            return {"error": str(exc)}

    async def _tool_analyze_resolution(self, args: dict) -> dict:
        """Delegate to the ResolutionAnalyzer for deep resolution analysis."""
        try:
            analyzer = _get_resolution_analyzer()
            result = await analyzer.analyze_market(
                market_id=args["market_id"],
                question=args["question"],
                description=args.get("description", ""),
                resolution_source=args.get("resolution_source", ""),
            )
            return result

        except ImportError:
            logger.warning("ResolutionAnalyzer not available")
            return {
                "error": "Resolution analyzer not yet available",
                "market_id": args.get("market_id"),
                "question": args.get("question"),
            }

        except Exception as exc:
            logger.error("analyze_resolution failed: %s", exc)
            return {"error": str(exc)}

    async def _tool_check_orderbook(self, args: dict) -> dict:
        """Check order book depth via the Polymarket CLOB API."""
        try:
            token_id = args["token_id"]
            async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
                resp = await client.get(
                    f"{CLOB_API_URL}/book",
                    params={"token_id": token_id},
                )
                if resp.status_code != 200:
                    return {
                        "error": f"Order book request failed (HTTP {resp.status_code})"
                    }

                book = resp.json()
                bids = book.get("bids", [])
                asks = book.get("asks", [])

                # Parse and summarize the order book
                bid_levels = [
                    {"price": float(b["price"]), "size": float(b["size"])}
                    for b in bids[:10]  # Top 10 levels
                ]
                ask_levels = [
                    {"price": float(a["price"]), "size": float(a["size"])}
                    for a in asks[:10]
                ]

                # Calculate key metrics
                best_bid = max((b["price"] for b in bid_levels), default=0.0)
                best_ask = min((a["price"] for a in ask_levels), default=1.0)
                spread = best_ask - best_bid if best_ask > best_bid else 0.0
                mid_price = (
                    (best_bid + best_ask) / 2 if (best_bid + best_ask) > 0 else 0.0
                )

                total_bid_liquidity = sum(b["price"] * b["size"] for b in bid_levels)
                total_ask_liquidity = sum(a["price"] * a["size"] for a in ask_levels)

                return {
                    "token_id": token_id,
                    "best_bid": best_bid,
                    "best_ask": best_ask,
                    "spread": round(spread, 4),
                    "spread_percent": round(spread / mid_price * 100, 2)
                    if mid_price > 0
                    else 0.0,
                    "mid_price": round(mid_price, 4),
                    "bid_levels": bid_levels[:5],  # Return top 5 for context
                    "ask_levels": ask_levels[:5],
                    "total_bid_liquidity_usd": round(total_bid_liquidity, 2),
                    "total_ask_liquidity_usd": round(total_ask_liquidity, 2),
                    "bid_depth": len(bids),
                    "ask_depth": len(asks),
                }

        except Exception as exc:
            logger.error("check_orderbook failed: %s", exc)
            return {"error": str(exc)}

    async def _tool_find_related_markets(self, args: dict) -> dict:
        """Find related markets on Polymarket via the Gamma API."""
        try:
            event_id = args.get("event_id")
            search_query = args.get("search_query")

            results: list[dict] = []

            async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
                # Strategy 1: Look up markets by event ID
                if event_id:
                    resp = await client.get(
                        f"{GAMMA_API_URL}/markets",
                        params={"event_id": event_id, "limit": 20},
                    )
                    if resp.status_code == 200:
                        for m in resp.json():
                            results.append(
                                {
                                    "condition_id": m.get("conditionId", ""),
                                    "question": m.get("question", ""),
                                    "outcome_prices": m.get("outcomePrices", []),
                                    "volume": m.get("volume", 0),
                                    "liquidity": m.get("liquidity", 0),
                                    "active": m.get("active", False),
                                    "slug": m.get("slug", ""),
                                }
                            )

                # Strategy 2: Text search for related markets
                if search_query:
                    resp = await client.get(
                        f"{GAMMA_API_URL}/markets",
                        params={
                            "active": "true",
                            "closed": "false",
                            "limit": 10,
                        },
                    )
                    if resp.status_code == 200:
                        query_lower = search_query.lower()
                        # Filter by keyword match in question
                        query_words = query_lower.split()
                        for m in resp.json():
                            question_lower = m.get("question", "").lower()
                            # Check if any significant query word appears
                            if any(
                                word in question_lower
                                for word in query_words
                                if len(word) > 3
                            ):
                                entry = {
                                    "condition_id": m.get("conditionId", ""),
                                    "question": m.get("question", ""),
                                    "outcome_prices": m.get("outcomePrices", []),
                                    "volume": m.get("volume", 0),
                                    "liquidity": m.get("liquidity", 0),
                                    "active": m.get("active", False),
                                    "slug": m.get("slug", ""),
                                }
                                # Avoid duplicates
                                if entry["condition_id"] not in {
                                    r["condition_id"] for r in results
                                }:
                                    results.append(entry)

            if not results:
                return {
                    "markets": [],
                    "message": "No related markets found",
                }

            return {
                "markets": results[:15],  # Cap at 15
                "count": len(results),
            }

        except Exception as exc:
            logger.error("find_related_markets failed: %s", exc)
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------

    def _build_system_prompt(
        self,
        market_question: Optional[str] = None,
        context: Optional[dict] = None,
    ) -> str:
        """Build the system prompt for the market analyzer agent."""
        prompt_parts = [
            "You are a Polymarket market analyst AI. Your job is to analyze "
            "prediction markets and provide comprehensive, data-driven insights "
            "to inform trading decisions.",
            "",
            "You have access to the following tools:",
            "- get_market_details: Fetch market info from Polymarket",
            "- search_news: Search for recent news and sentiment analysis",
            "- analyze_resolution: Deep analysis of resolution criteria",
            "- check_orderbook: Check order book depth and liquidity",
            "- find_related_markets: Find correlated or related markets",
            "",
            "GUIDELINES:",
            "1. Start by gathering relevant data using the tools before forming conclusions.",
            "2. Always check the order book if evaluating trade viability.",
            "3. Search for recent news to understand current sentiment.",
            "4. Analyze resolution criteria for edge cases or ambiguity risks.",
            "5. Look for related markets to detect correlations or arbitrage.",
            "6. Provide a clear, structured final analysis covering:",
            "   - Market overview and current state",
            "   - News and sentiment summary",
            "   - Resolution risk assessment",
            "   - Liquidity and order book analysis",
            "   - Related market correlations",
            "   - Overall recommendation with confidence level",
            "",
            "Be concise but thorough. Focus on actionable insights.",
            "Always state your confidence level (low/medium/high) and reasoning.",
        ]

        if market_question:
            prompt_parts.extend(
                [
                    "",
                    f"MARKET QUESTION: {market_question}",
                ]
            )

        if context:
            prompt_parts.extend(
                [
                    "",
                    "ADDITIONAL CONTEXT:",
                    json.dumps(context, indent=2, default=str),
                ]
            )

        return "\n".join(prompt_parts)

    # ------------------------------------------------------------------
    # Fallback: single-shot analysis when Agent framework is missing
    # ------------------------------------------------------------------

    async def _fallback_analyze(
        self,
        query: str,
        market_id: Optional[str],
        market_question: Optional[str],
        context: Optional[dict],
        model: Optional[str],
        session_id: str,
    ) -> dict:
        """Run a simpler single-shot analysis without the Agent loop.

        Manually gathers data from available tools, then asks the LLM
        to synthesize the findings in a single call.
        """
        tools_used: list[str] = []
        data_gathered: dict[str, Any] = {}

        # Gather market details if we have a market_id
        if market_id:
            market_data = await self._tool_get_market_details({"market_id": market_id})
            if "error" not in market_data:
                data_gathered["market_details"] = market_data
                tools_used.append("get_market_details")

        # Gather news
        news_query = market_question or query
        news_data = await self._tool_search_news(
            {"query": news_query, "max_results": 5}
        )
        if "error" not in news_data:
            data_gathered["news"] = news_data
            tools_used.append("search_news")

        # Synthesize with a single LLM call
        system_prompt = self._build_system_prompt(market_question, context)
        user_content = (
            f"Analyze this based on the gathered data:\n\n"
            f"Query: {query}\n\n"
            f"Data gathered:\n{json.dumps(data_gathered, indent=2, default=str)}"
        )

        try:
            manager = get_llm_manager()
            response = await manager.chat(
                messages=[
                    LLMMessage(role="system", content=system_prompt),
                    LLMMessage(role="user", content=user_content),
                ],
                model=model or "gpt-4o-mini",
                purpose="market_analysis",
            )
            analysis_text = response.content
        except Exception as exc:
            logger.error("Fallback LLM call failed: %s", exc)
            analysis_text = (
                f"Unable to generate LLM analysis: {exc}\n\n"
                f"Raw data gathered:\n{json.dumps(data_gathered, indent=2, default=str)}"
            )

        return {
            "query": query,
            "market_id": market_id,
            "analysis": analysis_text,
            "data_gathered": data_gathered,
            "tools_used": tools_used,
            "session_id": session_id,
        }

    # ------------------------------------------------------------------
    # Basic news search (fallback when NewsSentimentAnalyzer is missing)
    # ------------------------------------------------------------------

    async def _basic_news_search(self, query: str, max_results: int = 5) -> dict:
        """Minimal RSS-based news search without LLM sentiment analysis."""
        import urllib.parse
        from xml.etree import ElementTree

        try:
            encoded_query = urllib.parse.quote(query)
            url = (
                f"https://news.google.com/rss/search"
                f"?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
            )

            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(
                    url,
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                if response.status_code != 200:
                    return {
                        "articles": [],
                        "error": f"RSS fetch failed (HTTP {response.status_code})",
                    }

                root = ElementTree.fromstring(response.text)
                articles = []
                for item in root.findall(".//item")[:max_results]:
                    articles.append(
                        {
                            "title": item.findtext("title", ""),
                            "url": item.findtext("link", ""),
                            "published": item.findtext("pubDate", ""),
                            "source": item.findtext("source", ""),
                        }
                    )

                return {
                    "query": query,
                    "articles": articles,
                    "sentiment": {
                        "overall": "unknown",
                        "score": 0.0,
                        "confidence": 0.0,
                    },
                    "note": "Basic RSS fetch only -- no LLM sentiment analysis",
                }

        except Exception as exc:
            logger.error("Basic news search failed: %s", exc)
            return {"articles": [], "error": str(exc)}


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

market_analyzer = MarketAnalyzer()
