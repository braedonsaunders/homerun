"""Homerun MCP server — exposes backtest + iteration tools to external agents.

Two transports:

* **stdio**: ``python -m services.mcp.stdio_main`` — for agents that spawn
  the server as a subprocess (Claude Code, Cursor, Continue, etc.).

* **streamable HTTP** (``/mcp`` mounted on the main FastAPI app): for any
  agent that can reach the backend over HTTP.

Auth: none.  Homerun is a single-user, locally-run app; both transports
run open.  Bind the FastAPI server to localhost (the default) and the
MCP surface is implicitly local-only.

Tool surface: the existing ``services/ai/tools/`` AgentTool registry,
exposed verbatim.  No tool re-implementation.  Optional category-scoping
env vars (``HOMERUN_MCP_ALLOWED_CATEGORIES`` / ``HOMERUN_MCP_DENIED_
CATEGORIES``) let the operator opt out of exposing particular tool
categories during exploratory sessions.

The 9 missing tools the MCP surface needed (start/poll/stop param
iteration, walk-forward, drift report, recent opportunities,
backtest-run cache lookups) were added to the registry as
``services/ai/tools/iteration_tools.py`` — also visible to the
internal ReAct agent.
"""

from .server import build_mcp_server

__all__ = ["build_mcp_server"]
