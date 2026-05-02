"""MCP server factory — bridges the existing AgentTool registry to MCP.

There is NO duplication of tool implementations.  The platform has 70+
``AgentTool`` instances in ``services/ai/tools/`` (list_strategies,
get_strategy_details, run_strategy_backtest, start_param_iteration,
get_iteration_status, stop_iteration, run_walk_forward,
get_drift_report, etc.).  This module wraps each of them as an MCP
tool — same handler, same parameter schema — so external agents
(Claude Code, Cursor, Continue, etc.) talk to the same code paths the
internal ReAct agent uses.

Two transports:

* ``stdio_main.py`` builds this server and runs ``transport="stdio"``.
* ``http_app.py`` mounts ``streamable_http_app()`` at ``/mcp`` on the
  main FastAPI app.

Auth: none.  Homerun is a single-user, locally-run application; both
transports run open.  The optional category-scoping env vars below
are user-controlled — they let the operator opt OUT of exposing
particular tool categories (e.g. ``trading``) to MCP clients during
exploratory sessions.  They are not security boundaries.

Tool selection (env-var hooks, optional):

  ``HOMERUN_MCP_ALLOWED_CATEGORIES`` — csv whitelist.  When set, only
        tools in the listed categories are exposed.
  ``HOMERUN_MCP_DENIED_CATEGORIES``  — csv blocklist.  Takes
        precedence over the whitelist.

When neither is set, every registered AgentTool is exposed.
"""
from __future__ import annotations

import json
import os
from typing import Any

import mcp.types as mtypes
from mcp.server.lowlevel import Server

from services.ai.agent import AgentTool
from services.ai.tools import get_all_tools
from utils.logger import get_logger

logger = get_logger(__name__)


HOMERUN_MCP_INSTRUCTIONS = """\
Homerun MCP server — strategy backtesting + autonomous parameter iteration
for the homerun prediction-market trading platform.

Tools are grouped by category (strategies, backtest, iteration, diagnostics,
markets, portfolio, trading, wallets, news, analytics, system, cortex, data).
Use ``tools/list`` to see the full surface.

Typical workflow for "iterate strategy X until target_score Y":

  1. ``list_strategies``                    → find X (note the UUID/slug)
  2. ``get_strategy_details(strategy_id)``  → see the param schema + current config
  3. ``run_strategy_backtest(strategy_id, lookback_days=7)``  → baseline metrics
  4. ``start_param_iteration(strategy_id, target_score=Y, max_iterations=50)``
                                            → kicks off the LLM-driven loop;
                                              evaluator is the unified backtest
                                              with Cox fills + walk-forward gate
  5. ``get_iteration_status(strategy_id)``  → poll until target_reached=true or
                                              status="completed"
  6. ``run_walk_forward(strategy_id, config_overrides={...})``
                                            → optional overfit double-check on
                                              the kept params before going live

Strategies created or edited via MCP land at ``enabled=false``;
flipping them live still requires the trader-side switch in the UI.
"""


def _allowed_categories_from_env() -> set[str] | None:
    """Parse ``HOMERUN_MCP_ALLOWED_CATEGORIES`` into a set, or None."""
    raw = os.getenv("HOMERUN_MCP_ALLOWED_CATEGORIES", "").strip()
    if not raw:
        return None
    cats = {c.strip().lower() for c in raw.split(",") if c.strip()}
    return cats or None


def _denied_categories_from_env() -> set[str]:
    """Parse ``HOMERUN_MCP_DENIED_CATEGORIES`` into a set."""
    raw = os.getenv("HOMERUN_MCP_DENIED_CATEGORIES", "").strip()
    if not raw:
        return set()
    return {c.strip().lower() for c in raw.split(",") if c.strip()}


def _filter_tools(
    tools: dict[str, AgentTool],
    *,
    allowed_categories: set[str] | None,
    denied_categories: set[str],
) -> dict[str, AgentTool]:
    out: dict[str, AgentTool] = {}
    for name, tool in tools.items():
        cat = (getattr(tool, "category", "general") or "general").strip().lower()
        if cat in denied_categories:
            continue
        if allowed_categories is not None and cat not in allowed_categories:
            continue
        out[name] = tool
    return out


def _agent_tool_to_mcp_tool(tool: AgentTool) -> mtypes.Tool:
    """Convert an ``AgentTool`` into the ``mcp.types.Tool`` wire shape."""
    schema = tool.parameters or {"type": "object", "properties": {}}
    return mtypes.Tool(
        name=tool.name,
        description=tool.description or "",
        inputSchema=schema,
    )


def _coerce_result_to_text(result: Any) -> str:
    """Convert a handler's ``dict`` return into MCP TextContent text."""
    try:
        return json.dumps(result, default=str, ensure_ascii=False)
    except Exception:
        return str(result)


def build_mcp_server(*, name: str = "homerun") -> Server:
    """Return a configured MCP ``Server`` instance.

    Args:
        name: Server identity advertised to clients.

    The full AgentTool registry is exposed.  The operator can scope the
    surface via ``HOMERUN_MCP_ALLOWED_CATEGORIES`` /
    ``HOMERUN_MCP_DENIED_CATEGORIES`` env vars, but that's an opt-in
    convenience — there's no auth boundary, since the platform is
    single-user / locally-run.
    """
    server: Server = Server(name=name, instructions=HOMERUN_MCP_INSTRUCTIONS)

    allowed = _allowed_categories_from_env()
    denied = _denied_categories_from_env()

    # Resolve once at server-build time.  The agent-tool registry is
    # itself process-cached, so this is a fast dict lookup.
    available = get_all_tools()
    surface = _filter_tools(
        available,
        allowed_categories=allowed,
        denied_categories=denied,
    )
    logger.info(
        "MCP server '%s' surface: %d tools (allowed=%s, denied=%s)",
        name, len(surface),
        sorted(allowed) if allowed else "ALL",
        sorted(denied) if denied else [],
    )

    @server.list_tools()
    async def _list_tools() -> list[mtypes.Tool]:
        # ``surface`` is captured by closure; sorted by name for
        # deterministic ordering in the wire response.
        return [_agent_tool_to_mcp_tool(surface[k]) for k in sorted(surface.keys())]

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict[str, Any] | None) -> list[mtypes.ContentBlock]:
        tool = surface.get(name)
        if tool is None:
            return [mtypes.TextContent(
                type="text",
                text=json.dumps({"error": f"Unknown or filtered-out tool '{name}'"}),
            )]
        try:
            args = arguments or {}
            # Existing AgentTool handlers take a single ``dict`` and
            # return a ``dict`` (or list).  We just pass through.
            result = await tool.handler(args)
        except Exception as exc:
            logger.exception("MCP tool '%s' raised", name)
            return [mtypes.TextContent(
                type="text",
                text=json.dumps({"error": str(exc), "tool": name}),
            )]
        return [mtypes.TextContent(type="text", text=_coerce_result_to_text(result))]

    return server
