"""Worker module runner entrypoint.

Usage:
  python -m workers.runner workers.scanner_worker
"""

from __future__ import annotations

import asyncio
import importlib
import sys
from typing import Callable, Awaitable

if sys.platform == "win32" and hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


async def _run_worker(module_name: str) -> None:
    module = importlib.import_module(module_name)
    start_loop: Callable[[], Awaitable[None]] | None = getattr(module, "start_loop", None)
    if start_loop is None or not callable(start_loop):
        raise RuntimeError(f"Worker module '{module_name}' does not export callable start_loop()")
    await start_loop()


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m workers.runner workers.<module_name>")
    module_name = str(sys.argv[1]).strip()
    if not module_name.startswith("workers."):
        raise SystemExit(f"Invalid worker module '{module_name}'; expected 'workers.<name>'")
    asyncio.run(_run_worker(module_name))


if __name__ == "__main__":
    main()
