"""Homerun MCP server — exposes backtest + iteration tools to external agents.

Two transports:

* **stdio**: ``python -m services.mcp.stdio_main`` — for agents that spawn
  the server as a subprocess (Claude Code, Cursor, Continue, etc.).  No
  authentication; trusts the launching process.

* **streamable HTTP** (``/mcp`` mounted on the main FastAPI app): for remote
  agents.  Optional bearer-token auth via ``HOMERUN_MCP_API_KEY`` env var
  (or the ``mcp_api_key`` row in AppSettings) — when set the token must
  appear in the ``Authorization: Bearer <token>`` header.

Tool surface (12 tools, see ``services.mcp.tools``):

  Discovery     list_strategies, get_strategy, validate_strategy_source
  Backtest      run_backtest, get_backtest_run, list_backtest_runs,
                run_walk_forward
  Iteration     start_param_iteration, get_iteration_status, stop_iteration
  Diagnostics   get_drift_report, get_recent_opportunities

The tool implementations are thin async wrappers over existing services
(unified_runner, autoresearch_service, strategy_loader, etc.) so the same
code paths the UI uses also drive MCP-driven agent runs.
"""

from .server import build_mcp_server

__all__ = ["build_mcp_server"]
