"""HTTP/SSE MCP transport — mounted at ``/mcp`` on the main FastAPI app.

External agents (those that can reach the homerun backend over HTTP)
connect here.  The transport uses MCP's ``streamable-http`` (a single
HTTP endpoint that handles both POST requests and SSE responses), so
the URL the agent registers is just ``http(s)://<host>/mcp``.

Auth: bearer token via the ``HOMERUN_MCP_API_KEY`` env var.  When
unset the transport runs unauthenticated (loopback dev mode).  When
set, every request must carry ``Authorization: Bearer <key>``.

Safety: HTTP transport defaults to ``remote_safe_mode=True`` —
mutating-trade categories like ``trading`` are filtered out by
default.  Override via ``HOMERUN_MCP_ALLOWED_CATEGORIES`` /
``HOMERUN_MCP_DENIED_CATEGORIES``.

Usage from main.py::

    from services.mcp.http_app import mount_mcp_http
    mount_mcp_http(app)

The factory builds a shared FastMCP-based ``Server`` once at import
time and exposes both the streamable-http and the legacy SSE
mount paths.  Stateless mode means each tool call is self-contained;
the underlying iteration / experiment state lives in Postgres.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from .server import build_mcp_server, get_required_bearer_token

logger = logging.getLogger(__name__)


_MCP_MOUNT_PATH = "/mcp"


class _BearerTokenMiddleware(BaseHTTPMiddleware):
    """Enforce ``Authorization: Bearer <token>`` on the MCP sub-app.

    No-op when ``HOMERUN_MCP_API_KEY`` is unset (loopback dev mode).
    The token check uses constant-time comparison to avoid leaking
    timing information when an attacker probes for the right key.
    """

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        required = get_required_bearer_token()
        if required is None:
            return await call_next(request)

        # Allow OPTIONS preflight without a token so browsers can
        # negotiate CORS without a 401 round-trip.
        if request.method == "OPTIONS":
            return await call_next(request)

        auth_hdr = request.headers.get("authorization") or ""
        prefix = "Bearer "
        if not auth_hdr.startswith(prefix):
            return JSONResponse(
                {"error": "Missing Authorization: Bearer <token>"},
                status_code=status.HTTP_401_UNAUTHORIZED,
            )
        provided = auth_hdr[len(prefix):].strip()
        # Constant-time comparison — short-circuit on length mismatch
        # is fine because an attacker can already trivially observe
        # the expected length via a timing oracle on the env var.
        if not _safe_str_eq(provided, required):
            return JSONResponse(
                {"error": "Invalid bearer token"},
                status_code=status.HTTP_401_UNAUTHORIZED,
            )
        return await call_next(request)


def _safe_str_eq(a: str, b: str) -> bool:
    if len(a) != len(b):
        return False
    diff = 0
    for x, y in zip(a, b):
        diff |= ord(x) ^ ord(y)
    return diff == 0


def mount_mcp_http(parent_app: FastAPI) -> None:
    """Mount the MCP streamable-http sub-app at ``/mcp`` on ``parent_app``.

    Idempotent — calling twice is a no-op (we tag the app state).
    """
    if getattr(parent_app.state, "mcp_mounted", False):
        return

    # FastMCP's high-level server is the easiest path for HTTP
    # transport — it already provides a ``streamable_http_app()`` ASGI
    # subapp that handles both ``POST /mcp`` (JSON-RPC requests) and
    # ``GET /mcp`` (SSE response stream).  We use FastMCP here even
    # though stdio_main.py uses the lowlevel Server, because FastMCP
    # bundles the transport plumbing.  The tool registration path
    # mirrors server.build_mcp_server but goes through FastMCP's API.
    fastmcp_app = _build_fastmcp_subapp()

    # Route every request to /mcp/* through the bearer auth middleware
    # before it reaches the FastMCP ASGI subapp.
    parent_app.mount(_MCP_MOUNT_PATH, _wrap_with_auth(fastmcp_app))
    parent_app.state.mcp_mounted = True
    logger.info("MCP HTTP transport mounted at %s", _MCP_MOUNT_PATH)


def _build_fastmcp_subapp() -> ASGIApp:
    """Build the FastMCP HTTP sub-app exposing the AgentTool surface.

    Mirrors ``services.mcp.server.build_mcp_server`` but goes through
    FastMCP because its ``streamable_http_app()`` ships the transport
    plumbing.  Same tool surface — every AgentTool wrapped, same
    category filters, same stateless model.
    """
    import json

    import mcp.types as mtypes
    from mcp.server.fastmcp import FastMCP

    from services.ai.tools import get_all_tools
    from .server import (
        HOMERUN_MCP_INSTRUCTIONS,
        _allowed_categories_from_env,
        _denied_categories,
        _filter_tools,
    )

    mcp = FastMCP(
        name="homerun",
        instructions=HOMERUN_MCP_INSTRUCTIONS,
        stateless_http=True,
    )

    surface = _filter_tools(
        get_all_tools(),
        allowed_categories=_allowed_categories_from_env(),
        denied_categories=_denied_categories(remote_safe_mode=True),
    )

    # FastMCP introspects function signatures + docstrings to build
    # the JSON Schema it advertises.  Our handlers all take
    # ``(args: dict) -> dict`` so the introspection won't capture
    # individual fields.  We work around that by registering each
    # tool via the lower-level ``add_tool`` method with an explicit
    # name + description, and let the ``arguments`` collapse into
    # one object parameter — that's still valid JSON-RPC.

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
    """Register an AgentTool on a FastMCP server via add_tool().

    FastMCP wants a Python callable with type hints; we synthesize a
    closure with a ``**kwargs``-style signature so the JSON Schema
    falls back to ``inputSchema`` overrides registered after.
    """
    import json
    from mcp.server.fastmcp.tools import Tool as FastMCPTool

    handler = agent_tool.handler
    schema = agent_tool.parameters or {"type": "object", "properties": {}}

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

    # Build the FastMCP Tool directly so we can override the
    # parameters JSON Schema with the AgentTool's declared schema.
    # This bypasses pydantic introspection (which can't see our dict
    # contract) and uses the real per-tool schema instead.
    tool = FastMCPTool.from_function(
        fn=_wrapper,
        name=agent_tool.name,
        description=agent_tool.description or "",
    )
    # Replace the model-derived parameters with the declared schema.
    # FastMCP's Tool stores ``parameters`` as a dict — substitute it
    # in so list_tools() advertises the declared shape.
    tool.parameters = dict(schema)
    mcp._tool_manager._tools[agent_tool.name] = tool  # noqa: SLF001 — intentional


def _wrap_with_auth(asgi_app: ASGIApp) -> ASGIApp:
    """Wrap an ASGI app with the bearer-token auth middleware.

    We do this manually rather than using FastAPI's middleware chain
    because the FastMCP sub-app is a Starlette app mounted via
    ``parent_app.mount()`` and parent middlewares don't propagate
    into mounted sub-apps.
    """
    @asynccontextmanager
    async def _lifespan(scope, receive, send):
        await asgi_app(scope, receive, send)

    async def _wrapped(scope, receive, send):
        if scope["type"] != "http":
            return await asgi_app(scope, receive, send)

        required = get_required_bearer_token()
        if required is None:
            return await asgi_app(scope, receive, send)

        headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
        auth = headers.get("authorization", "")
        if scope.get("method", "").upper() == "OPTIONS":
            return await asgi_app(scope, receive, send)
        if not auth.startswith("Bearer "):
            await _send_unauth(send, "Missing Authorization: Bearer <token>")
            return
        provided = auth[len("Bearer "):].strip()
        if not _safe_str_eq(provided, required):
            await _send_unauth(send, "Invalid bearer token")
            return
        await asgi_app(scope, receive, send)

    return _wrapped


async def _send_unauth(send, message: str) -> None:
    body = (
        b'{"error":"' + message.encode("utf-8").replace(b'"', b"'") + b'"}'
    )
    await send({
        "type": "http.response.start",
        "status": 401,
        "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode("ascii")),
            (b"www-authenticate", b'Bearer realm="homerun-mcp"'),
        ],
    })
    await send({"type": "http.response.body", "body": body, "more_body": False})
