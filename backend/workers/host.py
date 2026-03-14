from __future__ import annotations

import argparse
import asyncio
import importlib
import logging
import os
import signal
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Optional

os.environ["HOMERUN_PROCESS_ROLE"] = "worker"

from config import RUNTIME_SETTINGS_PRECEDENCE, apply_runtime_settings_overrides
from models.database import AsyncSessionLocal
from models.model_registry import register_all_models
from services.event_bus import event_bus
from services.event_dispatcher import event_dispatcher
from services.intent_runtime import get_intent_runtime
from services.live_execution_service import live_execution_service
from services.market_cache import market_cache_service
from services.market_runtime import get_market_runtime
from services.position_monitor import position_monitor
from services.trader_orchestrator_state import read_orchestrator_snapshot
from services.traders_copy_trade_signal_service import traders_copy_trade_signal_service
from services.worker_state import list_worker_snapshots
from services.ws_feeds import get_feed_manager
from utils.logger import get_logger, setup_logging
from utils.utcnow import utcnow

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("NEWS_FAISS_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("EMBEDDING_DEVICE", "cpu")

register_all_models()
setup_logging(level=os.environ.get("LOG_LEVEL", "INFO"))
if not os.environ.get("HF_TOKEN"):
    logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)
logger = get_logger("workers.host")

_PLANE_CONFIGS: dict[str, dict[str, Any]] = {
    "all": {
        "worker_modules": (
            "workers.market_universe_worker",
            "workers.scanner_worker",
            "workers.scanner_slo_worker",
            "workers.tracked_traders_worker",
            "workers.discovery_worker",
            "workers.events_worker",
            "workers.news_worker",
            "workers.weather_worker",
            "workers.trader_reconciliation_worker",
            "workers.redeemer_worker",
        ),
        "runtime_names": (
            "trader_orchestrator",
            "trader_orchestrator_crypto",
        ),
        "load_strategy_registry": True,
        "load_data_source_registry": True,
        "start_event_bus": True,
        "start_event_dispatcher": True,
        "apply_runtime_settings": True,
        "start_intent_runtime": True,
        "start_feed_manager": True,
        "start_market_runtime": True,
        "load_market_cache": True,
        "load_news_feed": True,
        "initialize_live_execution": True,
        "start_copy_trade_service": True,
        "start_position_monitor": True,
        "start_fill_monitor": True,
    },
}


def _worker_name_from_module(module_name: str) -> str:
    return module_name.split(".")[-1].replace("_worker", "")


def _parse_iso_utc(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except Exception:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class WorkerHost:
    def __init__(self, plane_name: str) -> None:
        if plane_name not in _PLANE_CONFIGS:
            raise ValueError(f"Unsupported worker plane '{plane_name}'")
        self._plane_name = plane_name
        self._plane_config = dict(_PLANE_CONFIGS[plane_name])
        self._worker_modules = tuple(str(name) for name in self._plane_config.get("worker_modules", ()))
        self._runtime_names = tuple(str(name) for name in self._plane_config.get("runtime_names", ()))
        self._shutting_down = False
        self._stop_event = asyncio.Event()
        self._worker_tasks: dict[str, asyncio.Task] = {}
        self._worker_monitors: list[asyncio.Task] = []
        self._runtime_tasks: dict[str, asyncio.Task] = {}
        self._runtime_monitors: list[asyncio.Task] = []
        self._background_tasks: list[asyncio.Task] = []
        self._cpu_executor: ThreadPoolExecutor | None = None
        self._event_bus_started = False
        self._event_dispatcher_started = False
        self._intent_runtime_started = False
        self._feed_manager_started = False
        self._market_runtime_started = False
        self._copy_trade_service_started = False
        self._position_monitor_started = False
        self._fill_monitor_started = False

    def _enabled(self, key: str) -> bool:
        return bool(self._plane_config.get(key, False))

    async def _spawn_worker_task(self, module_name: str) -> asyncio.Task:
        module = importlib.import_module(module_name)
        start_loop = getattr(module, "start_loop", None)
        if start_loop is None:
            raise RuntimeError(f"{module_name} does not define start_loop()")
        task = asyncio.create_task(start_loop(), name=f"{self._plane_name}-{module_name.split('.')[-1]}")
        logger.info("Worker task started", plane=self._plane_name, worker=_worker_name_from_module(module_name))
        return task

    async def _cancel_worker_task(self, module_name: str, task: asyncio.Task) -> None:
        if task.done():
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.warning(
                "Worker task shutdown raised",
                plane=self._plane_name,
                worker=_worker_name_from_module(module_name),
                exc_info=exc,
            )

    async def _restart_worker_task(self, module_name: str, *, reason: str) -> None:
        if self._shutting_down:
            return
        current = self._worker_tasks.get(module_name)
        if current is not None:
            await self._cancel_worker_task(module_name, current)
        replacement = await self._spawn_worker_task(module_name)
        self._worker_tasks[module_name] = replacement
        logger.warning(
            "Worker task restarted",
            plane=self._plane_name,
            worker=_worker_name_from_module(module_name),
            reason=reason,
        )

    async def _monitor_worker_task(self, module_name: str) -> None:
        while not self._shutting_down:
            task = self._worker_tasks.get(module_name)
            if task is None:
                await asyncio.sleep(1.0)
                continue
            try:
                await task
                if self._shutting_down:
                    return
                if self._worker_tasks.get(module_name) is not task:
                    continue
                logger.error(
                    "Worker task exited unexpectedly",
                    plane=self._plane_name,
                    worker=_worker_name_from_module(module_name),
                )
                await asyncio.sleep(1.0)
                await self._restart_worker_task(module_name, reason="unexpected_exit")
            except asyncio.CancelledError:
                if self._shutting_down:
                    return
                if self._worker_tasks.get(module_name) is not task:
                    continue
                await asyncio.sleep(1.0)
                await self._restart_worker_task(module_name, reason="unexpected_cancel")
            except Exception as exc:
                if self._shutting_down:
                    return
                if self._worker_tasks.get(module_name) is not task:
                    continue
                logger.error(
                    "Worker task crashed",
                    plane=self._plane_name,
                    worker=_worker_name_from_module(module_name),
                    exc_info=exc,
                )
                await asyncio.sleep(1.0)
                await self._restart_worker_task(module_name, reason=f"unexpected_error:{type(exc).__name__}")

    async def _spawn_runtime_task(self, runtime_name: str) -> asyncio.Task:
        from workers import trader_orchestrator_worker as orchestrator_runtime

        if runtime_name == "trader_orchestrator":
            task = asyncio.create_task(
                orchestrator_runtime.start_loop(
                    lane="general",
                    notifier_enabled=True,
                    write_snapshot=True,
                ),
                name=f"{self._plane_name}-runtime-trader-orchestrator",
            )
        elif runtime_name == "trader_orchestrator_crypto":
            task = asyncio.create_task(
                orchestrator_runtime.start_loop(
                    lane="crypto",
                    notifier_enabled=False,
                    write_snapshot=False,
                ),
                name=f"{self._plane_name}-runtime-trader-orchestrator-crypto",
            )
        else:
            raise RuntimeError(f"Unsupported runtime task '{runtime_name}'")
        logger.info("Runtime task started", plane=self._plane_name, runtime=runtime_name)
        return task

    async def _cancel_runtime_task(self, runtime_name: str, task: asyncio.Task) -> None:
        if task.done():
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.warning("Runtime task shutdown raised", plane=self._plane_name, runtime=runtime_name, exc_info=exc)

    async def _restart_runtime_task(self, runtime_name: str, *, reason: str) -> None:
        if self._shutting_down:
            return
        current = self._runtime_tasks.get(runtime_name)
        if current is not None:
            await self._cancel_runtime_task(runtime_name, current)
        replacement = await self._spawn_runtime_task(runtime_name)
        self._runtime_tasks[runtime_name] = replacement
        logger.warning("Runtime task restarted", plane=self._plane_name, runtime=runtime_name, reason=reason)

    async def _monitor_runtime_task(self, runtime_name: str) -> None:
        while not self._shutting_down:
            task = self._runtime_tasks.get(runtime_name)
            if task is None:
                await asyncio.sleep(1.0)
                continue
            try:
                await task
                if self._shutting_down:
                    return
                if self._runtime_tasks.get(runtime_name) is not task:
                    continue
                logger.error("Runtime task exited unexpectedly", plane=self._plane_name, runtime=runtime_name)
                await asyncio.sleep(1.0)
                await self._restart_runtime_task(runtime_name, reason="unexpected_exit")
            except asyncio.CancelledError:
                if self._shutting_down:
                    return
                if self._runtime_tasks.get(runtime_name) is not task:
                    continue
                await asyncio.sleep(1.0)
                await self._restart_runtime_task(runtime_name, reason="unexpected_cancel")
            except Exception as exc:
                if self._shutting_down:
                    return
                if self._runtime_tasks.get(runtime_name) is not task:
                    continue
                logger.error("Runtime task crashed", plane=self._plane_name, runtime=runtime_name, exc_info=exc)
                await asyncio.sleep(1.0)
                await self._restart_runtime_task(runtime_name, reason=f"unexpected_error:{type(exc).__name__}")

    async def _monitor_worker_freshness(self) -> None:
        while not self._shutting_down:
            await asyncio.sleep(30.0)
            if self._shutting_down:
                return
            try:
                async with AsyncSessionLocal() as session:
                    snapshots = await list_worker_snapshots(session, include_stats=False)
                    orchestrator_snapshot = await read_orchestrator_snapshot(session)
            except Exception as exc:
                logger.warning("Worker freshness check failed", plane=self._plane_name, exc_info=exc)
                continue

            snapshot_by_name = {
                str(item.get("worker_name") or ""): item for item in snapshots if isinstance(item, dict)
            }
            now = utcnow()

            for module_name, task in list(self._worker_tasks.items()):
                if task.done():
                    continue
                worker_name = _worker_name_from_module(module_name)
                snapshot = snapshot_by_name.get(worker_name)
                if not snapshot:
                    continue
                updated_at = _parse_iso_utc(snapshot.get("updated_at"))
                if updated_at is None:
                    continue
                interval_seconds = max(1, int(snapshot.get("interval_seconds") or 60))
                stale_after_seconds = max(180, interval_seconds * 6)
                if worker_name == "scanner":
                    stale_after_seconds = max(stale_after_seconds, 360)
                elif worker_name == "tracked_traders":
                    stale_after_seconds = max(stale_after_seconds, 900)
                age_seconds = (now - updated_at).total_seconds()
                if age_seconds <= stale_after_seconds:
                    continue
                logger.error(
                    "Worker heartbeat stale; restarting worker task",
                    plane=self._plane_name,
                    worker=worker_name,
                    age_seconds=round(age_seconds, 1),
                    stale_after_seconds=stale_after_seconds,
                    current_activity=snapshot.get("current_activity"),
                )
                await self._restart_worker_task(module_name, reason="stale_heartbeat")

            orchestrator_runtime_snapshot = orchestrator_snapshot if isinstance(orchestrator_snapshot, dict) else None
            for runtime_name, task in list(self._runtime_tasks.items()):
                if task.done():
                    continue
                if runtime_name == "trader_orchestrator":
                    snapshot = orchestrator_runtime_snapshot
                    worker_name = "trader_orchestrator"
                else:
                    snapshot = snapshot_by_name.get("crypto")
                    worker_name = "crypto"
                if not snapshot:
                    continue
                updated_at = _parse_iso_utc(snapshot.get("updated_at"))
                if updated_at is None:
                    continue
                interval_seconds = max(1, int(snapshot.get("interval_seconds") or 5))
                stale_after_seconds = max(30, interval_seconds * 20) if worker_name == "crypto" else max(60, interval_seconds * 12)
                age_seconds = (now - updated_at).total_seconds()
                if age_seconds <= stale_after_seconds:
                    continue
                logger.error(
                    "Runtime heartbeat stale; restarting runtime task",
                    plane=self._plane_name,
                    runtime=runtime_name,
                    worker=worker_name,
                    age_seconds=round(age_seconds, 1),
                    stale_after_seconds=stale_after_seconds,
                    current_activity=snapshot.get("current_activity"),
                )
                await self._restart_runtime_task(runtime_name, reason="stale_heartbeat")

    async def _initialize_services(self) -> None:
        logger.info("Worker plane database pool ready", plane=self._plane_name)

        from models.database import start_pool_watchdog

        self._background_tasks.append(start_pool_watchdog())

        if self._enabled("load_strategy_registry"):
            try:
                from services.opportunity_strategy_catalog import ensure_all_strategies_seeded
                from services.strategy_loader import strategy_loader

                async with AsyncSessionLocal() as session:
                    seeded = await ensure_all_strategies_seeded(session)
                loaded = await strategy_loader.refresh_all_from_db()
                logger.info(
                    "Strategy registries loaded",
                    plane=self._plane_name,
                    seeded=seeded.get("seeded", 0),
                    loaded=len(loaded.get("loaded", [])),
                    errors=len(loaded.get("errors", {})),
                )
            except Exception as exc:
                logger.warning("Failed to preload strategy registries", plane=self._plane_name, exc_info=exc)

        if self._enabled("load_data_source_registry"):
            try:
                from services.data_source_catalog import ensure_all_data_sources_seeded
                from services.data_source_loader import data_source_loader

                async with AsyncSessionLocal() as session:
                    seeded = await ensure_all_data_sources_seeded(session)
                loaded = await data_source_loader.refresh_all_from_db()
                logger.info(
                    "Data source registries loaded",
                    plane=self._plane_name,
                    seeded=seeded.get("seeded", 0),
                    loaded=len(loaded.get("loaded", [])),
                    errors=len(loaded.get("errors", {})),
                )
            except Exception as exc:
                logger.warning("Failed to preload data source registries", plane=self._plane_name, exc_info=exc)

        if self._enabled("start_event_bus"):
            await event_bus.start()
            self._event_bus_started = True
        if self._enabled("start_event_dispatcher"):
            await event_dispatcher.start()
            self._event_dispatcher_started = True

        if self._enabled("apply_runtime_settings"):
            try:
                await apply_runtime_settings_overrides()
                logger.info(
                    "Runtime settings overrides applied",
                    plane=self._plane_name,
                    precedence=RUNTIME_SETTINGS_PRECEDENCE,
                )
            except Exception as exc:
                logger.warning("Failed to apply runtime settings overrides", plane=self._plane_name, exc_info=exc)

        if self._enabled("start_intent_runtime"):
            await get_intent_runtime().start()
            self._intent_runtime_started = True

        if self._enabled("start_feed_manager"):
            feed_manager = get_feed_manager()
            if not getattr(feed_manager, "_started", False):
                await feed_manager.start()
                self._feed_manager_started = True

        if self._enabled("start_market_runtime"):
            await get_market_runtime().start()
            self._market_runtime_started = True

        if self._enabled("load_market_cache"):
            try:
                await market_cache_service.load_from_db()
            except Exception as exc:
                logger.warning("Market cache load failed", plane=self._plane_name, exc_info=exc)

        if self._enabled("load_news_feed"):
            try:
                from services.news.feed_service import news_feed_service

                await news_feed_service.load_from_db()
            except Exception as exc:
                logger.warning("News feed preload failed", plane=self._plane_name, exc_info=exc)

        if self._enabled("initialize_live_execution"):
            try:
                trading_initialized = await live_execution_service.initialize()
                if trading_initialized:
                    logger.info("Live execution service initialized", plane=self._plane_name)
                else:
                    logger.info("Live execution service not initialized - credentials not configured", plane=self._plane_name)
            except Exception as exc:
                logger.warning("Live execution initialization failed", plane=self._plane_name, exc_info=exc)

        if self._enabled("start_copy_trade_service"):
            await traders_copy_trade_signal_service.start()
            await traders_copy_trade_signal_service.refresh_scope()
            self._copy_trade_service_started = True

        if self._enabled("start_position_monitor"):
            await position_monitor.start()
            self._position_monitor_started = True

        if self._enabled("start_fill_monitor"):
            try:
                from services.fill_monitor import fill_monitor

                await fill_monitor.start()
                self._fill_monitor_started = True
            except Exception as exc:
                logger.warning("Fill monitor start failed", plane=self._plane_name, exc_info=exc)

    async def _start_plane(self) -> None:
        loop = asyncio.get_running_loop()
        cpu_count = os.cpu_count() or 4
        self._cpu_executor = ThreadPoolExecutor(
            max_workers=max(cpu_count * 2 + 8, 16),
            thread_name_prefix=f"{self._plane_name}-cpu-pool",
        )
        loop.set_default_executor(self._cpu_executor)

        def _global_exception_handler(loop_ref, context):
            exc = context.get("exception")
            message = context.get("message", "Unhandled asyncio exception")
            if exc is not None:
                logger.error(
                    "Unhandled asyncio exception",
                    plane=self._plane_name,
                    context_message=message,
                    exc_info=exc,
                )
            else:
                logger.error(
                    "Unhandled asyncio exception",
                    plane=self._plane_name,
                    context_message=message,
                )

        loop.set_exception_handler(_global_exception_handler)

        await self._initialize_services()

        for module_name in self._worker_modules:
            self._worker_tasks[module_name] = await self._spawn_worker_task(module_name)
            self._worker_monitors.append(
                asyncio.create_task(
                    self._monitor_worker_task(module_name),
                    name=f"{self._plane_name}-monitor-{module_name.split('.')[-1]}",
                )
            )

        for runtime_name in self._runtime_names:
            self._runtime_tasks[runtime_name] = await self._spawn_runtime_task(runtime_name)
            self._runtime_monitors.append(
                asyncio.create_task(
                    self._monitor_runtime_task(runtime_name),
                    name=f"{self._plane_name}-monitor-{runtime_name}",
                )
            )

        freshness_task = asyncio.create_task(
            self._monitor_worker_freshness(),
            name=f"{self._plane_name}-monitor-freshness",
        )
        self._worker_monitors.append(freshness_task)

        logger.info(
            "Worker plane started",
            plane=self._plane_name,
            worker_count=len(self._worker_tasks),
            runtime_count=len(self._runtime_tasks),
        )

    async def _stop_plane(self) -> None:
        self._shutting_down = True

        for runtime_name, task in list(self._runtime_tasks.items()):
            await self._cancel_runtime_task(runtime_name, task)
        for task in self._runtime_monitors:
            if not task.done():
                task.cancel()
        for task in self._runtime_monitors:
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass

        for module_name, task in list(self._worker_tasks.items()):
            await self._cancel_worker_task(module_name, task)
        for task in self._worker_monitors:
            if not task.done():
                task.cancel()
        for task in self._worker_monitors:
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass

        if self._fill_monitor_started:
            try:
                from services.fill_monitor import fill_monitor

                fill_monitor.stop()
            except Exception:
                pass
            self._fill_monitor_started = False
        if self._position_monitor_started:
            position_monitor.stop()
            self._position_monitor_started = False
        if self._copy_trade_service_started:
            traders_copy_trade_signal_service.stop()
            self._copy_trade_service_started = False
        if self._market_runtime_started:
            await get_market_runtime().stop()
            self._market_runtime_started = False
        if self._feed_manager_started:
            await get_feed_manager().stop()
            self._feed_manager_started = False
        if self._intent_runtime_started:
            await get_intent_runtime().stop()
            self._intent_runtime_started = False
        if self._event_dispatcher_started:
            await event_dispatcher.stop()
            self._event_dispatcher_started = False
        if self._event_bus_started:
            await event_bus.stop()
            self._event_bus_started = False

        for task in self._background_tasks:
            task.cancel()
        for task in self._background_tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass

        if self._cpu_executor is not None:
            self._cpu_executor.shutdown(wait=False, cancel_futures=True)
            self._cpu_executor = None

    async def run(self) -> None:
        await self._start_plane()
        loop = asyncio.get_running_loop()

        def _request_stop() -> None:
            self._stop_event.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _request_stop)
            except (NotImplementedError, RuntimeError, ValueError):
                pass

        await self._stop_event.wait()

    async def shutdown(self) -> None:
        await self._stop_plane()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Homerun worker plane")
    parser.add_argument("plane", choices=sorted(_PLANE_CONFIGS.keys()))
    return parser.parse_args()


async def _run(plane_name: str) -> None:
    host = WorkerHost(plane_name)
    try:
        await host.run()
    finally:
        await host.shutdown()


def main() -> None:
    args = _parse_args()
    asyncio.run(_run(str(args.plane)))


if __name__ == "__main__":
    main()
