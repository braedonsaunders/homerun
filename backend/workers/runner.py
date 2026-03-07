"""Worker module runner entrypoint.

Usage:
  python -m workers.runner workers.scanner_worker

Each worker subprocess is supervised by the main process, which restarts
it on unexpected exit.  This runner adds a second layer of resilience:
if `start_loop()` raises an unexpected exception, the runner logs the
crash, disposes the DB pool (so stale connections don't carry over),
and re-enters `start_loop()` after a back-off delay.

After MAX_RAPID_CRASHES within RAPID_CRASH_WINDOW_SECONDS the runner
gives up and exits, letting the main-process supervisor handle the
restart with its own back-off.
"""

from __future__ import annotations

import asyncio
import importlib
import sys
import time
from typing import Callable, Awaitable


MAX_RAPID_CRASHES = 5
RAPID_CRASH_WINDOW_SECONDS = 60.0
RESTART_BASE_DELAY_SECONDS = 2.0
RESTART_MAX_DELAY_SECONDS = 30.0


async def _run_worker(module_name: str) -> None:
    module = importlib.import_module(module_name)
    start_loop: Callable[[], Awaitable[None]] | None = getattr(module, "start_loop", None)
    if start_loop is None or not callable(start_loop):
        raise RuntimeError(f"Worker module '{module_name}' does not export callable start_loop()")

    # Import lazily so the logger and DB pool are configured by the time we use them.
    from utils.logger import get_logger
    from models.database import start_pool_watchdog
    logger = get_logger(f"worker.runner.{module_name.split('.')[-1]}")

    # Start the pool watchdog in this worker subprocess too.
    start_pool_watchdog()

    crash_times: list[float] = []
    consecutive_crashes = 0

    while True:
        try:
            logger.info("Starting worker loop", module=module_name)
            await start_loop()
            # start_loop() returned normally — this means the worker
            # decided to stop (e.g. shutdown signal).  Exit cleanly.
            logger.info("Worker loop returned normally", module=module_name)
            return
        except asyncio.CancelledError:
            logger.info("Worker cancelled", module=module_name)
            return
        except SystemExit:
            raise
        except KeyboardInterrupt:
            return
        except Exception as exc:
            consecutive_crashes += 1
            now = time.monotonic()
            crash_times.append(now)
            # Trim to window
            crash_times[:] = [t for t in crash_times if now - t < RAPID_CRASH_WINDOW_SECONDS]

            logger.error(
                "Worker crashed; will restart",
                module=module_name,
                error_type=type(exc).__name__,
                error=str(exc)[:500],
                consecutive_crashes=consecutive_crashes,
                crashes_in_window=len(crash_times),
                exc_info=exc,
            )

            if len(crash_times) >= MAX_RAPID_CRASHES:
                logger.error(
                    "Worker exceeded max rapid crashes (%d in %.0fs); exiting for supervisor restart",
                    MAX_RAPID_CRASHES,
                    RAPID_CRASH_WINDOW_SECONDS,
                    module=module_name,
                )
                sys.exit(1)

            # Dispose the DB pool so stale/leaked connections don't carry over.
            try:
                from models.database import recover_pool
                await recover_pool()
            except Exception:
                pass

            delay = min(
                RESTART_BASE_DELAY_SECONDS * (2 ** (consecutive_crashes - 1)),
                RESTART_MAX_DELAY_SECONDS,
            )
            logger.info(
                "Restarting worker after %.1fs delay",
                delay,
                module=module_name,
            )
            await asyncio.sleep(delay)

            # Re-import the module in case the crash was caused by stale state.
            try:
                module = importlib.reload(module)
                start_loop = getattr(module, "start_loop", None)
                if start_loop is None or not callable(start_loop):
                    logger.error("Module reload lost start_loop; exiting", module=module_name)
                    sys.exit(1)
            except Exception as reload_exc:
                logger.error(
                    "Module reload failed; using original module",
                    module=module_name,
                    error=str(reload_exc)[:200],
                )


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m workers.runner workers.<module_name>")
    module_name = str(sys.argv[1]).strip()
    if not module_name.startswith("workers."):
        raise SystemExit(f"Invalid worker module '{module_name}'; expected 'workers.<name>'")
    asyncio.run(_run_worker(module_name))


if __name__ == "__main__":
    main()
