"""stdio MCP entrypoint.

Run via:

    python -m services.mcp.stdio_main

Or register with Claude Code:

    claude mcp add homerun -- python -m services.mcp.stdio_main

stdio transport trusts the launching process — no bearer-token check.
The server connects to the same Postgres + Redis that the live
backend uses, so backtests + iterations run against real data.

Environment knobs:

  HOMERUN_MCP_ALLOWED_CATEGORIES  csv whitelist (e.g. "backtest,iteration,
                                  diagnostics,strategies").  Omit to expose
                                  every registered tool.
  HOMERUN_MCP_DENIED_CATEGORIES   csv blocklist; takes precedence over
                                  whitelist.

stdio defaults to the FULL surface — including ``trading`` and
``cortex`` mutating tools — because the launching CLI is trusted.
For remote / HTTP transport those categories default to denied.
"""
from __future__ import annotations

import asyncio
import logging
import sys

from .server import build_mcp_server

logger = logging.getLogger("homerun.mcp.stdio")


async def _run_async() -> None:
    """Build the server and run it on the stdio transport."""
    from mcp.server.stdio import stdio_server

    server = build_mcp_server(name="homerun", remote_safe_mode=False)

    init_options = server.create_initialization_options()

    async with stdio_server() as (read_stream, write_stream):
        logger.info("homerun MCP stdio server starting")
        await server.run(read_stream, write_stream, init_options)


def main() -> int:
    # Quiet down noisy startup imports — stdio piping is sensitive to
    # garbage on stderr being interpreted by some clients.  Route logs
    # to stderr at WARNING+ only.
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        stream=sys.stderr,
    )
    try:
        asyncio.run(_run_async())
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # pragma: no cover
        logger.exception("homerun MCP stdio server crashed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
