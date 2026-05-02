"""
Agent Tool Registry.

Central registry for all tools available to LLM agents.  Each sub-module
exposes a ``build_tools() -> list[AgentTool]`` function.  The registry
collects them into a single flat mapping keyed by tool name.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from services.ai.agent import AgentTool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy builder — populated on first call, then cached.
# ---------------------------------------------------------------------------

_tool_cache: dict[str, "AgentTool"] | None = None


def _build_all_tools() -> dict[str, "AgentTool"]:
    """Import every tool module and collect AgentTool instances."""
    from services.ai.tools import (
        analytics_tools,
        cortex_tools,
        data_tools,
        iteration_tools,
        market_tools,
        news_tools,
        portfolio_tools,
        strategy_tools,
        system_tools,
        trading_tools,
        wallet_tools,
        web_tools,
    )

    modules = [
        market_tools,
        portfolio_tools,
        trading_tools,
        wallet_tools,
        strategy_tools,
        news_tools,
        analytics_tools,
        web_tools,
        system_tools,
        cortex_tools,
        data_tools,
        # Strategy-scoped param iteration + advanced backtest tools.
        # Provides start/poll/stop iteration, walk-forward, drift report,
        # recent opportunities, and run_id-keyed backtest cache lookups —
        # the surface the MCP server bridge surfaces to external agents.
        iteration_tools,
    ]

    tool_map: dict[str, "AgentTool"] = {}
    for mod in modules:
        try:
            for tool in mod.build_tools():
                if tool.name in tool_map:
                    logger.warning("Duplicate tool name %r from %s — skipping", tool.name, mod.__name__)
                    continue
                tool_map[tool.name] = tool
        except Exception:
            logger.exception("Failed to build tools from %s", mod.__name__)

    logger.info("Agent tool registry loaded %d tools", len(tool_map))
    return tool_map


def get_all_tools() -> dict[str, "AgentTool"]:
    """Return the full ``{name: AgentTool}`` mapping (cached)."""
    global _tool_cache
    if _tool_cache is None:
        _tool_cache = _build_all_tools()
    return _tool_cache


def resolve_tools(tool_names: list[str]) -> list["AgentTool"]:
    """Resolve a list of tool name strings into AgentTool instances."""
    all_tools = get_all_tools()
    resolved = []
    for name in tool_names:
        tool = all_tools.get(name)
        if tool:
            resolved.append(tool)
        else:
            logger.warning("Unknown tool requested: %s", name)
    return resolved


def get_available_tools_metadata() -> list[dict]:
    """Return metadata dicts for every registered tool (for the UI)."""
    all_tools = get_all_tools()
    result = []
    for tool in all_tools.values():
        result.append({
            "name": tool.name,
            "description": tool.description,
            "category": getattr(tool, "category", "general"),
            "parameters": tool.parameters,
        })
    return sorted(result, key=lambda t: (t["category"], t["name"]))


def invalidate_cache() -> None:
    """Force rebuild of the tool cache (e.g. after hot-reload)."""
    global _tool_cache
    _tool_cache = None
