"""HTTP/SSE MCP transport — mounted at ``/mcp`` on the main FastAPI app.

External agents (those that can reach the homerun backend over HTTP)
connect here.  The transport uses MCP's ``streamable-http`` (a single
HTTP endpoint that handles both POST requests and SSE responses), so
the URL the agent registers is just ``http(s)://<host>/mcp``.

Auth: none.  Homerun is a single-user, locally-run application.  Bind
the FastAPI server to localhost (the default) and the MCP surface is
implicitly local-only.  If you expose the backend on a public
interface, put a real reverse proxy in front of it — that's an
infrastructure concern, not an MCP concern.

The category-scoping env vars (``HOMERUN_MCP_ALLOWED_CATEGORIES`` /
``HOMERUN_MCP_DENIED_CATEGORIES``, see ``server.py``) are user
controls — the operator can opt out of exposing particular tool
categories during exploratory sessions.

Usage from main.py::

    from services.mcp.http_app import mount_mcp_http
    mount_mcp_http(app)
"""
from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import FastAPI
from starlette.types import ASGIApp

from services.ai.tools import get_all_tools

from .server import (
    HOMERUN_MCP_INSTRUCTIONS,
    _allowed_categories_from_env,
    _denied_categories_from_env,
    _filter_tools,
)

logger = logging.getLogger(__name__)


_MCP_MOUNT_PATH = "/mcp"


def mount_mcp_http(parent_app: FastAPI) -> None:
    """Mount the MCP streamable-http sub-app at ``/mcp`` on ``parent_app``.

    Idempotent — calling twice is a no-op (we tag the app state).
    """
    if getattr(parent_app.state, "mcp_mounted", False):
        return

    fastmcp_app = _build_fastmcp_subapp()
    parent_app.mount(_MCP_MOUNT_PATH, fastmcp_app)
    parent_app.state.mcp_mounted = True
    logger.info("MCP HTTP transport mounted at %s", _MCP_MOUNT_PATH)


def _build_fastmcp_subapp() -> ASGIApp:
    """Build the FastMCP HTTP sub-app exposing the AgentTool surface.

    Mirrors ``services.mcp.server.build_mcp_server`` but goes through
    FastMCP because its ``streamable_http_app()`` ships the transport
    plumbing.  Same tool surface — every AgentTool wrapped, same
    category filters, same stateless model.
    """
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP(
        name="homerun",
        instructions=HOMERUN_MCP_INSTRUCTIONS,
        stateless_http=True,
    )

    surface = _filter_tools(
        get_all_tools(),
        allowed_categories=_allowed_categories_from_env(),
        denied_categories=_denied_categories_from_env(),
    )

    for tool_name in sorted(surface.keys()):
        agent_tool = surface[tool_name]
        _register_agent_tool_on_fastmcp(mcp, agent_tool)

    logger.info(
        "FastMCP HTTP sub-app built with %d tools (allowed=%s)",
        len(surface),
        sorted(_allowed_categories_from_env() or []) or "ALL",
    )
    return mcp.streamable_http_app()


def _register_agent_tool_on_fastmcp(mcp: Any, agent_tool: Any) -> None:
    """Register an AgentTool on a FastMCP server.

    FastMCP wants a Python callable with type hints; we synthesize a
    closure with a ``**kwargs``-style signature and override the
    advertised JSON Schema with the AgentTool's declared one — so
    FastMCP's pydantic introspection (which can't see our dict
    contract) gets superseded by the real per-tool schema.
    """
    from mcp.server.fastmcp.tools import Tool as FastMCPTool

    handler = agent_tool.handler

    async def _wrapper(**kwargs: Any) -> str:
        try:
            result = await handler(dict(kwargs))
        except Exception as exc:
            logger.exception("MCP HTTP tool '%s' raised", agent_tool.name)
            return json.dumps({"error": str(exc), "tool": agent_tool.name})
        try:
            return json.dumps(result, default=str, ensure_ascii=False)
        except Exception:
            return str(result)

    tool = FastMCPTool.from_function(
        fn=_wrapper,
        name=agent_tool.name,
        description=agent_tool.description or "",
    )
    # Replace the model-derived parameters with the declared schema.
    # FastMCP's Tool stores ``parameters`` as a dict — substitute it
    # in so list_tools() advertises the declared shape.
    tool.parameters = dict(agent_tool.parameters or {"type": "object", "properties": {}})
    mcp._tool_manager._tools[agent_tool.name] = tool  # noqa: SLF001 — intentional
