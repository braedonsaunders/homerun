from __future__ import annotations

import argparse
import asyncio
import traceback
import importlib
import logging
import os
import signal
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

os.environ["HOMERUN_PROCESS_ROLE"] = "worker"
# Plane is set in main() once the CLI arg is parsed; downstream
# services consult ``HOMERUN_WORKER_PLANE`` to gate plane-specific
# behavior (e.g. only the ``trading`` plane runs the Polymarket
# user-channel WS feed — running it in two processes against the
# same wallet causes the server to drop both connections).

from config import RUNTIME_SETTINGS_PRECEDENCE, apply_runtime_settings_overrides, settings
from models.database import AsyncSessionLocal
from models.model_registry import register_all_models
from services.event_bus import event_bus
from services.event_dispatcher import event_dispatcher
from services.intent_runtime import get_intent_runtime
from services.live_execution_service import live_execution_service
from services.market_cache import market_cache_service
from services.market_runtime import get_market_runtime
from services.position_monitor import position_monitor
from services.runtime_status import runtime_status
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
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

register_all_models()
# Optional debug log file — when ``HOMERUN_DEBUG_LOG_FILE`` (or the
# per-plane override ``HOMERUN_DEBUG_LOG_FILE_<PLANE>``) is set,
# every log record is also persisted to that file (in addition to
# stdout).  The launcher's ``-Debug`` flag sets this per-plane so a
# perf monitoring harness (``tools/perf_harness.py``) can tail the
# JSON output without having to spawn its own copy of the worker
# stack.  Empty / unset → stdout-only as before.
def _resolve_debug_log_file() -> str | None:
    plane_arg_index = (sys.argv.index("--plane") + 1) if "--plane" in sys.argv else 1
    plane_token = sys.argv[plane_arg_index] if len(sys.argv) > plane_arg_index else ""
    plane_token = (plane_token or "").strip().lower()
    if plane_token:
        per_plane = os.environ.get(f"HOMERUN_DEBUG_LOG_FILE_{plane_token.upper()}", "").strip()
        if per_plane:
            return per_plane
    generic = os.environ.get("HOMERUN_DEBUG_LOG_FILE", "").strip()
    return generic or None
_debug_log_file = _resolve_debug_log_file()
setup_logging(level=os.environ.get("LOG_LEVEL", "INFO"), log_file=_debug_log_file)
if not os.environ.get("HF_TOKEN"):
    logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)
logger = get_logger("workers.host")

# Tracemalloc is OFF by default — the snapshot phase scales with
# the number of live blocks, and a worker carrying tens of millions
# of small objects (post-leak) takes 20+ minutes to dump, defeating
# the purpose of a real-time diagnostic.  Set ``HOMERUN_TRACE_MEMORY=1``
# to opt in for targeted leak hunts; the file-trigger dump still
# works without it (psutil + GC + module-singleton sections are
# always available, and tell us the singleton-leak shape directly).
if os.environ.get("HOMERUN_TRACE_MEMORY") == "1":
    from utils.memory_diagnostic import start_tracemalloc

    start_tracemalloc()

if os.name == "nt":
    import msvcrt
else:
    import fcntl

_LOCKS_DIR = Path(__file__).resolve().parents[1] / ".runtime" / "locks"

# Plane separation: the trading hot path and the ML-heavy news/weather
# ingest workers run in separate processes so a 2GB sentence-
# transformers + FAISS heap can never page out the live-money trader.
# Production deploys ``trading`` and ``news`` as two distinct host
# processes; ``all`` is kept only for legacy single-process use.
_PLANE_CONFIGS: dict[str, dict[str, Any]] = {
    "trading": {
        "worker_modules": (
            # Scanner + detection workers (market_universe, scanner,
            # scanner_slo, search_index, events) live on the ``detection``
            # plane.  Their full-snapshot scan + catalog refresh are CPU/loop
            # heavy, and they hand opportunities to trading purely via the DB
            # ``trade_signals`` write path (the intent_runtime bridge + Redis
            # signal bridge) — the same cross-plane handoff news/discovery use
            # — so they never need to share the orchestrator's event loop.
            # The detection plane runs its OWN feed_manager + intent_runtime so
            # scanner WS-strict signals (required_token_ids) still get live
            # quotes + project.  See the ``detection`` plane below.
            "workers.trader_reconciliation_worker",
            "workers.fast_trader_runtime",
            # Isolated fast stop-loss / exit-risk loop (own fast DB pool +
            # WS-tick + 2s sweep). First-class risk control that can't be
            # starved by the heavyweight reconcile — the fix for the
            # 2026-05 full-notional soak losses.
            "workers.exit_risk_worker",
            "workers.redeemer_worker",
            # Keep the in-process latency / empirical-constants / Cox-
            # model caches warm so the order hot path never blocks on
            # a DB round trip.  Cheap (~few SELECTs per minute).
            "workers.fill_simulator_refresh_worker",
        ),
        "runtime_names": (
            "trader_orchestrator",
            "trader_orchestrator_crypto",
        ),
        "load_strategy_registry": True,
        # Only load strategies the trading plane actually runs.  Excluding
        # ``news`` and ``weather`` is critical: ``news_edge`` strategy
        # imports ``services.news.semantic_matcher`` at module top, which
        # transitively imports ``sentence_transformers`` → ``transformers``
        # → ``torch`` (~2 GB heap).  Without this filter the trading plane
        # carries the ML stack despite never matching news/weather.
        "strategy_source_keys": ("scanner", "traders", "crypto", "sports", "manual"),
        "load_data_source_registry": True,
        "start_event_bus": True,
        "start_event_dispatcher": True,
        "apply_runtime_settings": True,
        "start_intent_runtime": True,
        "start_feed_manager": True,
        "start_market_runtime": True,
        "load_market_cache": True,
        "load_news_feed": False,
        "initialize_live_execution": True,
        "start_copy_trade_service": True,
        "start_position_monitor": True,
        "start_fill_monitor": True,
        # Recording moved to its own ``recording`` plane (process isolation
        # of the broad WS book stream + parquet encoding from the trading
        # event loop).  OFF here so we never double-run two recording pools.
        "start_recording": False,
    },
    "news": {
        # ML/embedding workers.  Hand off opportunities to the trading
        # plane via DB writes (``bridge_opportunities_to_signals``);
        # no shared in-process state with trading.
        "worker_modules": (
            "workers.news_worker",
            "workers.weather_worker",
        ),
        "runtime_names": (),
        "load_strategy_registry": True,
        # News plane owns the ML strategies.  ``news_edge`` lives here
        # (sentence-transformers + FAISS) along with weather strategies
        # that subscribe to ``WeatherDataEvent``.  See trading plane
        # comment for the rationale.
        "strategy_source_keys": ("news", "weather"),
        "load_data_source_registry": True,
        "start_event_bus": True,
        "start_event_dispatcher": True,
        "apply_runtime_settings": True,
        "start_intent_runtime": False,
        "start_feed_manager": False,
        "start_market_runtime": False,
        "load_market_cache": False,
        "load_news_feed": True,
        "initialize_live_execution": False,
        "start_copy_trade_service": False,
        "start_position_monitor": False,
        "start_fill_monitor": False,
    },
    "discovery": {
        # Wallet discovery + smart-pool / confluence work.  Both call
        # Polymarket REST in fan-out patterns that, when colocated with
        # the trader orchestrator, produce 90+ in-flight ``try_connect``
        # tasks and 5-8 s event-loop stalls (5 h soak, 11/2026/05).
        # Isolating them here trades a small amount of cross-process
        # event handling (``TradersConfluenceStrategy`` runs in this
        # plane and its opportunities reach trading via
        # ``bridge_opportunities_to_signals`` → DB
        # ``trade_signals``) for keeping the trading event loop
        # responsive.  No live-execution / intent_runtime / feed_manager
        # — this plane only reads market data from DB and writes
        # opportunities back to DB.
        "worker_modules": (
            "workers.discovery_worker",
            "workers.tracked_traders_worker",
            # Drains the ProviderImportJob queue (polybacktest etc.).
            # Pure REST I/O fan-out — same shape as the other discovery
            # workers; share the plane to keep the trading event loop
            # clean.  Also runs the strategy reverse-engineer queue
            # since both are bounded long-running operator-initiated
            # jobs that benefit from the same plane.
            "workers.provider_import_worker",
            "workers.strategy_reverse_engineer_worker",
            # Drains the BacktestRun job queue.  Backtests are CPU-
            # heavy (1M-snapshot replays chew the GIL for minutes) so
            # they MUST live off the trading plane — running them
            # here gives full process isolation from the live
            # orchestrator.  See workers/backtest_worker.py.
            "workers.backtest_worker",
            # Cox PH fill-model trainer (lifelines + pandas + scipy).  It's
            # backtest-fidelity infrastructure — the trader hot path only
            # READS the persisted ``fill_probability_models`` active row — so
            # it belongs with the other backtest workers here, decoupled from
            # the news subsystem toggle and kept off the trading plane's RAM.
            # The worker itself skips the refit when the model is still fresh,
            # so colocating it here adds no startup cost on a warm model.
            "workers.cox_trainer_worker",
        ),
        "runtime_names": (),
        # Load the ``traders`` strategy bucket so
        # ``TradersConfluenceStrategy`` is registered with this plane's
        # dispatcher — without it ``tracked_traders_worker``'s
        # ``trader_activity`` event would have no subscribers.  The
        # strategy is pure-CPU (no live state), and the resulting
        # opportunities are bridged to DB so the trading plane still
        # picks them up.
        "load_strategy_registry": True,
        "strategy_source_keys": ("traders",),
        "load_data_source_registry": True,
        "start_event_bus": True,
        "start_event_dispatcher": True,
        "apply_runtime_settings": True,
        "start_intent_runtime": False,
        "start_feed_manager": False,
        "start_market_runtime": False,
        # Do not bulk-load ``cached_markets`` in discovery. The table can
        # be hundreds of MB and discovery only needs occasional point
        # lookups; MarketCacheService handles those lazily by primary key.
        "load_market_cache": False,
        "load_news_feed": False,
        "initialize_live_execution": False,
        "start_copy_trade_service": False,
        "start_position_monitor": False,
        "start_fill_monitor": False,
    },
    "all": {
        # Legacy: single-process mode bundling trading + news.
        # Prefer ``trading`` + ``news`` as separate processes in prod.
        "worker_modules": (
            "workers.market_universe_worker",
            "workers.scanner_worker",
            "workers.scanner_slo_worker",
            "workers.search_index_worker",
            "workers.tracked_traders_worker",
            "workers.discovery_worker",
            "workers.events_worker",
            "workers.news_worker",
            "workers.weather_worker",
            "workers.cox_trainer_worker",
            "workers.trader_reconciliation_worker",
            "workers.fast_trader_runtime",
            "workers.redeemer_worker",
            "workers.fill_simulator_refresh_worker",
            "workers.provider_import_worker",
            "workers.strategy_reverse_engineer_worker",
            # Drains the BacktestRun job queue.  Backtests are CPU-
            # heavy (1M-snapshot replays chew the GIL for minutes) so
            # they MUST live off the trading plane — running them
            # here gives full process isolation from the live
            # orchestrator.  See workers/backtest_worker.py.
            "workers.backtest_worker",
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
        # Legacy single-process mode records inline.
        "start_recording": True,
    },
    "recording": {
        # Dedicated microstructure-recording plane.  Runs the ISOLATED
        # RecordingFeedManager pool + LiveMarketDataIngestor + BookParquetSink
        # and the recording-session manager — moved off the trading plane so
        # the broad WS book stream, parquet encoding, and provider_datasets
        # catalog UPSERT can never load the orchestrator's event loop (the
        # "recording must not affect the orchestrator" invariant, now enforced
        # by a process boundary).  Reads the shared market-catalog file +
        # Redis recording config; writes parquet + the small provider_datasets
        # UPSERT.  No strategies / feed / orchestrator / live-execution.
        "worker_modules": (),
        "runtime_names": (),
        "load_strategy_registry": False,
        "strategy_source_keys": (),
        "load_data_source_registry": False,
        "start_event_bus": False,
        "start_event_dispatcher": False,
        "apply_runtime_settings": False,
        "start_intent_runtime": False,
        "start_feed_manager": False,
        "start_market_runtime": False,
        "load_market_cache": False,
        "load_news_feed": False,
        "initialize_live_execution": False,
        "start_copy_trade_service": False,
        "start_position_monitor": False,
        "start_fill_monitor": False,
        "start_recording": True,
    },
    "detection": {
        # Scanner + opportunity-detection plane.  Runs the full-snapshot
        # scanner, catalog refresh, SLO, search-index, and events workers off
        # the trading event loop.  Opportunities reach trading via the DB
        # ``trade_signals`` write path (intent_runtime bridge + Redis signal
        # bridge), exactly like the news/discovery planes.  Runs its OWN
        # feed_manager + intent_runtime so scanner WS-strict signals
        # (required_token_ids) get live book quotes and project to trade_signals
        # — the scanner subscribes its candidate tokens to this plane's feed and
        # intent_runtime subscribes any missing ones.  No orchestrator /
        # live-execution / recording here.
        "worker_modules": (
            "workers.market_universe_worker",
            "workers.scanner_worker",
            "workers.scanner_slo_worker",
            "workers.search_index_worker",
            "workers.events_worker",
        ),
        "runtime_names": (),
        "load_strategy_registry": True,
        "strategy_source_keys": ("scanner",),
        "load_data_source_registry": True,
        "start_event_bus": True,
        "start_event_dispatcher": True,
        "apply_runtime_settings": True,
        "start_intent_runtime": True,
        "start_feed_manager": True,
        "start_market_runtime": False,
        "load_market_cache": False,
        "load_news_feed": False,
        "initialize_live_execution": False,
        "start_copy_trade_service": False,
        "start_position_monitor": False,
        "start_fill_monitor": False,
        "start_recording": False,
    },
    "reconciliation": {
        # Cold reconciliation plane (A.3).  Runs ONLY trader_reconciliation_worker
        # in DETECTION-ONLY mode (the worker reads HOMERUN_WORKER_PLANE ==
        # "reconciliation" -> place_exits=False): the heavy per-candidate sweep +
        # settlement / terminal-audit / bulk-reconcile bookkeeping runs here on
        # its OWN process so the trading event loop is never loaded by it.  It
        # NEVER places/cancels live orders — live_execution is initialized in
        # READ-ONLY mode (initialize_live_execution=True + live_execution_read_only
        # =True): it loads CLOB creds so authenticated READS (order snapshots,
        # wallet/balance) succeed, but every venue mutation (place/cancel/allowance)
        # is hard-blocked at the service. Combined with the reconcile place_exits
        # gate + dual-writer scoping, the cold plane reads + bookkeeps but mutates
        # nothing and never persists the pending-exit execution lifecycle. The
        # trading plane keeps exit_risk_loop (primary, real-time exits) + a
        # place_exits=True reconcile backstop at a longer cadence. Strategies are
        # loaded so the cold pass computes the same exit decisions (telemetry);
        # no feed / intent_runtime / market_runtime here.
        "worker_modules": (
            "workers.trader_reconciliation_worker",
        ),
        "runtime_names": (),
        "load_strategy_registry": True,
        "strategy_source_keys": ("scanner", "traders", "crypto", "sports", "manual"),
        "load_data_source_registry": True,
        "start_event_bus": True,
        "start_event_dispatcher": False,
        "apply_runtime_settings": True,
        "start_intent_runtime": False,
        "start_feed_manager": False,
        "start_market_runtime": False,
        "load_market_cache": True,
        "load_news_feed": False,
        # Init live_execution but READ-ONLY: creds for authenticated reads;
        # every venue mutation (place/cancel/allowance) hard-blocked.
        "initialize_live_execution": True,
        "live_execution_read_only": True,
        "start_copy_trade_service": False,
        "start_position_monitor": False,
        "start_fill_monitor": False,
        "start_recording": False,
    },
}


def _worker_name_from_module(module_name: str) -> str:
    return module_name.split(".")[-1].replace("_worker", "")


def get_plane_worker_summary() -> dict[str, dict[str, Any]]:
    """Public, runtime-stable view of the plane -> workers/runtimes/services
    structure for the GUI home page + the /health/planes endpoint.  Single
    source of truth = _PLANE_CONFIGS, so the GUI never duplicates the plane
    layout.  Excludes the legacy ``all`` single-process plane (never spawned).
    ``services`` are the notable plane-level background subsystems derived from
    the feature flags — what actually runs when a plane has no worker_modules
    (e.g. the recording plane)."""
    service_flags: tuple[tuple[str, str], ...] = (
        ("start_recording", "recorder"),
        ("start_intent_runtime", "intent_runtime"),
        ("start_feed_manager", "feed_manager"),
        ("start_market_runtime", "market_runtime"),
        ("initialize_live_execution", "live_execution"),
        ("start_copy_trade_service", "copy_trade"),
        ("start_position_monitor", "position_monitor"),
        ("start_fill_monitor", "fill_monitor"),
        ("load_news_feed", "news_feed"),
    )
    out: dict[str, dict[str, Any]] = {}
    for plane, cfg in _PLANE_CONFIGS.items():
        if plane == "all":
            continue
        out[plane] = {
            "workers": [_worker_name_from_module(m) for m in cfg.get("worker_modules", ())],
            "runtimes": [str(r) for r in cfg.get("runtime_names", ())],
            "services": [label for flag, label in service_flags if cfg.get(flag)],
        }
    return out


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


def _should_suppress_asyncio_exception(message: str, exc: Exception | None) -> bool:
    if "Proactor" in message or "_loop_reading" in message or "_loop_writing" in message:
        return True
    if isinstance(exc, asyncio.InvalidStateError) and "protocol.data_received()" in message:
        return True
    if isinstance(exc, ConnectionError) and "unexpected connection_lost() call" in str(exc):
        return True
    if isinstance(exc, AttributeError) and "backend_pid" in str(exc):
        return True
    # Post-cancel cleanup noise from the asyncpg/SQLAlchemy stack.
    # These surface as "Task exception was never retrieved" from the
    # session's fire-and-forget drain/invalidate path after a wait_for
    # timeout cancels a query.  The connection has already been
    # invalidated by the pool listener — re-logging the cleanup-side
    # error at ERROR level is just noise.
    exc_type_name = type(exc).__name__ if exc is not None else ""
    exc_module = type(exc).__module__ if exc is not None else ""
    if exc_type_name == "InternalClientError" and exc_module.startswith("asyncpg"):
        return True
    if exc_type_name == "ResourceClosedError" and "sqlalchemy" in exc_module:
        return True
    if exc_type_name == "InterfaceError" and "sqlalchemy" in exc_module:
        # rvf5 is the SQLAlchemy stable error code for the
        # "connection is closed" family surfaced mid-cleanup.  Real
        # interface errors on user-facing paths are caught and logged
        # by the calling code; the ones that reach the global handler
        # come exclusively from drain tasks after cancellation.
        msg_text = f"{message} {exc}".lower()
        if "rvf5" in msg_text or "connection is closed" in msg_text:
            return True
    return False


class _WorkerPlaneLock:
    def __init__(self, plane_name: str) -> None:
        self._plane_name = plane_name
        self._path = _LOCKS_DIR / f"worker-plane.{plane_name}.lock"
        self._handle = None

    def acquire(self) -> None:
        _LOCKS_DIR.mkdir(parents=True, exist_ok=True)
        handle = self._path.open("a+", encoding="utf-8")
        try:
            handle.seek(0)
            handle.write("0")
            handle.flush()
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            handle.seek(0)
            handle.truncate()
            handle.write(str(os.getpid()))
            handle.flush()
        except OSError:
            handle.close()
            raise RuntimeError(f"Worker plane '{self._plane_name}' is already running")
        except Exception:
            handle.close()
            raise
        self._handle = handle

    def release(self) -> None:
        handle = self._handle
        self._handle = None
        if handle is None:
            return
        try:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


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
        self._restart_grace: dict[str, float] = {}  # module_name -> monotonic time of last restart
        # Crash history: module_name -> list of monotonic crash timestamps
        # within the last 1h (older entries pruned on access). Used to:
        #   * surface a structured "high crash rate" log when a worker is
        #     thrashing — without this the operator only sees one
        #     "Worker task crashed" line per crash, with no easy way to
        #     spot frequency
        #   * escalate restart cooldown so a thrashing worker doesn't
        #     hammer its dependencies
        self._crash_history: dict[str, list[float]] = {}
        self._plane_lock = _WorkerPlaneLock(self._plane_name)

    def _enabled(self, key: str) -> bool:
        return bool(self._plane_config.get(key, False))

    def _schedule_background_task(self, coro: Awaitable[Any], *, name: str) -> asyncio.Task[Any]:
        task = asyncio.create_task(coro, name=name)
        self._background_tasks.append(task)
        return task

    def _schedule_background_startup(
        self,
        *,
        task_name: str,
        starter: Callable[[], Awaitable[Any]],
        failure_message: str,
        started_attr: str | None = None,
        started_check: Callable[[], bool] | None = None,
    ) -> None:
        async def _run() -> None:
            try:
                if started_check is not None and started_check():
                    return
                await starter()
                if started_attr is not None:
                    setattr(self, started_attr, True)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(failure_message, plane=self._plane_name, exc_info=exc)

        self._schedule_background_task(_run(), name=f"{self._plane_name}-{task_name}")

    async def _run_trade_signal_pruner_loop(self) -> None:
        """Periodic pruner: deletes terminal trade_signals past the
        reactivation lookback.  Runs every 5 minutes on the trading
        plane only.

        Pre-2026-04-30: 30-minute cadence + 24h horizon. Under load,
        terminal signals accumulated faster than they were pruned —
        ``list_unconsumed_trade_signals`` slowed to 2-4s/cycle (vs
        the fast trader's 3s budget) and lock contention on the
        per-row UPDATE spiked to 4s.  Tightened the cadence to 5 min
        and kept the 24h horizon (the reactivation window for
        skipped signals; shorter would lose recoverable state).
        """
        # Stagger initial fire so we don't compete with startup queries.
        await asyncio.sleep(60.0)
        while not self._shutting_down:
            try:
                from services.maintenance import maintenance_service

                result = await maintenance_service.cleanup_terminal_trade_signals(
                    older_than_hours=24,
                )
                deleted = int(result.get("trade_signals_deleted", 0) or 0)
                if deleted > 0:
                    logger.info(
                        "Pruned terminal trade signals",
                        plane=self._plane_name,
                        deleted=deleted,
                    )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "Terminal trade signal pruner cycle failed",
                    plane=self._plane_name,
                    exc_info=exc,
                )
            try:
                await asyncio.sleep(300.0)  # 5 minutes
            except asyncio.CancelledError:
                raise

    async def _run_orchestrator_slo_monitor_loop(self) -> None:
        """Hot-loop SLO alarm (Phase D).

        Reads the trader orchestrator's per-lane cycle lag from in-process
        runtime_status and raises a throttled operator alert when the lag
        exceeds the configured SLA.  The threshold lives in the orchestrator
        control's ``global_runtime.orchestrator_cycle_slo_seconds`` (default
        30s; 0 disables) so it is UI-configurable, never a hidden constant.

        The lag itself comes from in-process runtime_status (no DB); only the
        SLA threshold is refreshed from the control row, once per poll.
        Trading plane only.
        """
        import time as _time
        from services.notifier_bridge import publish_alert
        from services.trader_orchestrator_state import read_orchestrator_control

        _POLL_SECONDS = 10.0
        _ALERT_THROTTLE_SECONDS = 300.0
        last_alert_mono: dict[str, float] = {}

        # Let the orchestrator warm up before judging cycle lag.
        try:
            await asyncio.sleep(30.0)
        except asyncio.CancelledError:
            return

        while not self._shutting_down:
            try:
                slo_seconds: Optional[float] = None
                db_snapshots: list = []
                _last_decision_at = None
                try:
                    async with AsyncSessionLocal() as session:
                        control = await read_orchestrator_control(session, read_only=True)
                        try:
                            from sqlalchemy import text as _sql_text
                            db_snapshots = list((await session.execute(_sql_text(
                                "SELECT id, last_run_at, current_activity "
                                "FROM trader_orchestrator_snapshot WHERE running = true"
                            ))).mappings().all())
                        except Exception:
                            db_snapshots = []
                        try:
                            _last_decision_at = (await session.execute(_sql_text(
                                "SELECT max(created_at) FROM trader_decision_checks "
                                "WHERE created_at > now() - interval '15 minutes'"
                            ))).scalar()
                        except Exception:
                            _last_decision_at = None
                    global_runtime = (control.get("settings") or {}).get("global_runtime") or {}
                    raw_slo = global_runtime.get("orchestrator_cycle_slo_seconds")
                    slo_seconds = float(raw_slo) if raw_slo is not None else None
                except Exception:
                    slo_seconds = None

                # Reliable stall capture from the PERSISTED snapshot's last_run_at
                # (the real cycle-completion time). A hung cycle keeps the
                # in-process heartbeat fresh but never advances last_run_at, so the
                # runtime_status-based check below misses it entirely — which is
                # exactly why an obviously-stalled orchestrator produced no dump.
                _stall_threshold = max((slo_seconds or 30.0) * 3.0, 90.0)
                _now_mono2 = _time.monotonic()
                # Decisions still landing within the stall window => the
                # orchestrator IS working; the manage-only maintenance cycle just
                # runs at a longer cadence than the SLO threshold and lags its
                # snapshot. Suppress the false STALL alarm + await-stack dump then
                # (a TRUE stall stops producing decisions, so this still fires).
                _decisions_fresh = False
                if _last_decision_at is not None:
                    _ld = _last_decision_at
                    if getattr(_ld, "tzinfo", None) is None:
                        _ld = _ld.replace(tzinfo=timezone.utc)
                    _decisions_fresh = (utcnow() - _ld).total_seconds() <= _stall_threshold
                for _snap in db_snapshots:
                    if _decisions_fresh:
                        continue
                    _lr = _snap.get("last_run_at")
                    if _lr is None:
                        continue
                    if getattr(_lr, "tzinfo", None) is None:
                        _lr = _lr.replace(tzinfo=timezone.utc)
                    _db_lag = (utcnow() - _lr).total_seconds()
                    if _db_lag <= _stall_threshold:
                        continue
                    _sid = str(_snap.get("id") or "?")
                    if (_now_mono2 - last_alert_mono.get(f"dbstall:{_sid}", 0.0)) < _ALERT_THROTTLE_SECONDS:
                        continue
                    last_alert_mono[f"dbstall:{_sid}"] = _now_mono2
                    logger.warning(
                        "Orchestrator DB-snapshot STALL — cycle not completing",
                        snapshot_id=_sid,
                        lag_seconds=round(_db_lag, 0),
                        activity=_snap.get("current_activity"),
                    )
                    self._dump_orchestrator_await_stacks(_sid, _db_lag)

                if slo_seconds is not None and slo_seconds > 0:
                    now_mono = _time.monotonic()
                    for lane in ("general", "crypto"):
                        snapshot = runtime_status.get_orchestrator(lane)
                        if not isinstance(snapshot, dict) or not snapshot.get("running"):
                            continue
                        last_run_raw = snapshot.get("last_run_at")
                        last_run = _parse_iso_utc(last_run_raw) if last_run_raw else None
                        if last_run is None:
                            continue
                        lag_seconds = (utcnow() - last_run).total_seconds()
                        if lag_seconds <= slo_seconds:
                            continue
                        if now_mono - last_alert_mono.get(lane, 0.0) < _ALERT_THROTTLE_SECONDS:
                            continue
                        last_alert_mono[lane] = now_mono
                        activity = snapshot.get("current_activity") or "unknown"
                        logger.warning(
                            "Orchestrator SLO breach",
                            plane=self._plane_name,
                            lane=lane,
                            lag_seconds=round(lag_seconds, 1),
                            slo_seconds=slo_seconds,
                            activity=activity,
                        )
                        # On a SEVERE stall (cycle minutes behind, not merely
                        # slow), dump the orchestrator's await-stack so the exact
                        # suspended line is captured. The lag alarm says the
                        # cycle is parked; only the stack says WHERE — and that's
                        # what's needed to fix the ingest-lag (stale-fill) stall.
                        if lag_seconds > max(slo_seconds * 3.0, 60.0):
                            self._dump_orchestrator_await_stacks(lane, lag_seconds)
                        try:
                            await publish_alert(
                                f"Orchestrator SLO breach ({lane}): cycle lag "
                                f"{lag_seconds:.1f}s > SLA {slo_seconds:.0f}s — trading loop "
                                f"under resource pressure (activity: {activity})",
                                category="operator",
                            )
                        except Exception:
                            pass
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.debug("Orchestrator SLO monitor cycle failed", exc_info=exc)

            try:
                await asyncio.sleep(_POLL_SECONDS)
            except asyncio.CancelledError:
                raise

    def _dump_orchestrator_await_stacks(self, lane: str, lag_seconds: float) -> None:
        """Dump the await-stack of every asyncio task parked in orchestrator
        code, so a multi-minute cycle stall is captured with the exact line it
        is suspended on. ``task.print_stack`` on a suspended coroutine yields
        the await chain — which the lag alarm alone cannot provide."""
        import io
        import inspect as _inspect

        try:
            tasks = list(asyncio.all_tasks())
        except Exception:
            return
        # task.get_stack() only returns the TOP frame for these coroutines, so
        # walk the cr_await chain by hand: coro.cr_frame gives the suspended
        # line, coro.cr_await gives what it's awaiting (a sub-coroutine to
        # descend into, or a Task/Future — and a Task we descend into via
        # get_coro()). That reconstructs the real deep chain, including the
        # shielded inner task that the cycle is actually parked on.
        buf = io.StringIO()
        buf.write(
            f"=== ORCH STALL cr_await walk lane={lane} lag={lag_seconds:.0f}s tasks={len(tasks)} ===\n"
        )
        for task in tasks:
            try:
                chain: list[str] = []
                obj = task.get_coro()
                seen: set[int] = set()
                steps = 0
                while obj is not None and steps < 80:
                    steps += 1
                    if id(obj) in seen:
                        break
                    seen.add(id(obj))
                    frame = (
                        getattr(obj, "cr_frame", None)
                        or getattr(obj, "gi_frame", None)
                        or getattr(obj, "ag_frame", None)
                    )
                    if frame is not None:
                        co = frame.f_code
                        chain.append(f"{co.co_filename}:{frame.f_lineno} {co.co_name}")
                    nxt = (
                        getattr(obj, "cr_await", None)
                        or getattr(obj, "gi_yieldfrom", None)
                        or getattr(obj, "ag_await", None)
                    )
                    if nxt is None:
                        break
                    if _inspect.iscoroutine(nxt) or _inspect.isgenerator(nxt):
                        obj = nxt
                        continue
                    nm = nxt.get_name() if hasattr(nxt, "get_name") else ""
                    chain.append(f"<await {type(nxt).__name__} {nm}>")
                    gc = getattr(nxt, "get_coro", None)
                    if callable(gc):
                        try:
                            obj = gc()
                            continue
                        except Exception:
                            break
                    break
                # Skip trivial idle tasks (one frame parked on a sleep/future).
                if len(chain) < 2 and not any("trader_orchestrator" in c for c in chain):
                    continue
                buf.write(f"\n--- task={task.get_name()} depth={len(chain)} ---\n")
                for c in chain:
                    buf.write(f"  {c}\n")
            except Exception:
                continue
        logger.warning("%s", buf.getvalue())

    async def _run_skeleton_signal_retention_loop(self) -> None:
        """Periodic sweep that DELETEs orphaned `trade_signals`
        skeleton rows on the discovery plane.

        Plan 0010's publish-side skeleton-INSERT commits a placeholder
        ``(source, dedupe_key)`` row BEFORE the projection loop
        enriches it.  If publish dies mid-call, the row stays in
        ``trade_signals`` with ``payload_json IS NULL`` and
        ``runtime_sequence IS NULL``, and is invisible to the
        terminal-row pruner (which keys on ``expires_at < now()``).
        Plan 0011 stamps a defensive ``expires_at = now + ttl`` on
        the skeleton AND adds this loop, which deletes skeletons
        older than ``max_age_seconds`` outright.  Discovery-plane
        only — keeps the orphan-deletion path off the trader-cycle's
        10 s budget.

        Default cadence: 15 min.  Default age threshold: 1 hour.
        """
        interval_seconds = max(
            60,
            int(getattr(settings, "INTENT_RUNTIME_SKELETON_RETENTION_INTERVAL_SECONDS", 900) or 900),
        )
        max_age_seconds = max(
            60,
            int(getattr(settings, "INTENT_RUNTIME_SKELETON_RETENTION_MAX_AGE_SECONDS", 3600) or 3600),
        )
        # Stagger initial fire so we don't compete with startup queries.
        await asyncio.sleep(60.0)
        while not self._shutting_down:
            try:
                from services.skeleton_signal_retention import prune_stuck_skeletons

                async with AsyncSessionLocal() as session:
                    deleted = await prune_stuck_skeletons(
                        session,
                        max_age_seconds=max_age_seconds,
                    )
                # Log every sweep at INFO so the operator can correlate
                # with publish failures.  Steady-state ``deleted=0`` is
                # the healthy case; non-zero rows mean the publish path
                # is dying mid-call somewhere upstream.
                logger.info(
                    "Stuck-skeleton retention sweep",
                    plane=self._plane_name,
                    deleted=deleted,
                    max_age_seconds=max_age_seconds,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "Stuck-skeleton retention sweep failed",
                    plane=self._plane_name,
                    exc_info=exc,
                )
            try:
                await asyncio.sleep(float(interval_seconds))
            except asyncio.CancelledError:
                raise

    async def _run_recording_session_loop(self) -> None:
        """Tick the recording-session manager every 5s.

        Promotes ``scheduled`` sessions whose start time has passed,
        refreshes ``rows_captured`` for ``running`` sessions, and
        auto-terminates them at ``scheduled_end_at`` /
        ``max_duration_seconds``.  Cheap — bulk SELECT + small UPDATE
        per running session — so 5s cadence is fine.
        """
        # Stagger initial fire so we don't compete with startup queries.
        await asyncio.sleep(15.0)
        while not self._shutting_down:
            try:
                from services.recording_session_service import session_loop_tick

                await session_loop_tick()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # Classify transient DB-pressure errors so they don't
                # masquerade as a real worker failure in the soak log
                # — observed during the 2026-05-09 13:30:34 / 13:31:04
                # cascade where the trading-plane pool was momentarily
                # exhausted. Recording-session ticks are non-critical;
                # the next tick (5 s away) sees a recovered pool.
                err_text = str(exc).lower()
                err_type = type(exc).__name__.lower()
                is_transient_db = (
                    isinstance(exc, (asyncio.TimeoutError, TimeoutError))
                    or "cannot switch to state" in err_text
                    or "connection is closed" in err_text
                    or "another operation" in err_text
                    or "connectiondoesnotexisterror" in err_type
                    or "interfaceerror" in err_type
                    or "queryCanceled" in err_type.lower()
                )
                if is_transient_db:
                    logger.info(
                        "Recording session loop tick skipped under DB pressure",
                        plane=self._plane_name,
                        error_type=type(exc).__name__,
                    )
                else:
                    logger.warning(
                        "Recording session loop cycle failed",
                        plane=self._plane_name,
                        exc_info=exc,
                    )
            try:
                await asyncio.sleep(5.0)
            except asyncio.CancelledError:
                raise

    async def _run_recorder_subscription_loop(self) -> None:
        """Run the proactive recorder subscription service.

        Periodically reads MarketCatalog and subscribes the WebSocket
        feed to the top liquid markets so the microstructure recorder
        has book data for every market a strategy might fire on.
        See ``services.recorder_subscription_service`` for the policy.
        """
        try:
            from services.market_data_ingestor import get_market_data_ingestor
            from services.recorder_subscription_service import run_loop
            from services.recording_feed import get_recording_feed_manager

            # Start the microstructure ingestor (book/trade buffering + parquet
            # sink + catalog flush loops) HERE — it was previously started by the
            # trading feed_manager, but recording now lives on this plane.  The
            # RecordingFeedManager pool's record_book / record_trade callbacks
            # feed it; without this start() the buffered books would never flush
            # to parquet.  Idempotent.
            get_market_data_ingestor().start()
            # Start the ISOLATED recording WS pool (its own connections + cache,
            # decoupled from the trading feed) before the recorder begins
            # subscribing the broad set onto it.  The ingestor's record_book /
            # record_trade callbacks are bound to the pool's cache, so book/trade
            # persistence rides these connections — never the trading-critical
            # feed the orchestrator reads prices + fires stop-losses from.
            await get_recording_feed_manager().start()
            await run_loop()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Recorder subscription loop crashed",
                plane=self._plane_name,
                exc_info=exc,
            )

    async def _initialize_live_execution_background(self) -> None:
        # Retry until ready.  A single attempt is not enough: a transient at
        # startup (DB pool contention, a slow credential/funder/CLOB round-trip)
        # leaves live_execution not-ready, and the orchestrator's live cycle
        # then defers every cycle (run_worker_loop's is_ready() gate) — so
        # without retrying here an arming user would wait forever.  Exponential
        # backoff keeps a genuinely unconfigured account (no credentials) from
        # spinning, while still self-healing a transient and picking up
        # credentials configured at runtime.
        backoff = 2.0
        max_backoff = 30.0
        # Read-only planes (cold reconciliation) load creds for authenticated
        # READS but must never mutate the venue. Set this BEFORE any init so the
        # service comes up with all mutations hard-blocked.
        live_execution_service.set_read_only(self._enabled("live_execution_read_only"))
        while not self._shutting_down:
            try:
                if live_execution_service.is_ready():
                    return
                trading_initialized = await live_execution_service.initialize()
                if trading_initialized:
                    logger.info("Live execution service initialized", plane=self._plane_name)
                    return
                logger.info(
                    "Live execution service not initialized (credentials not configured); will retry",
                    plane=self._plane_name,
                    last_error=live_execution_service.get_last_init_error(),
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "Live execution initialization failed; will retry",
                    plane=self._plane_name,
                    exc_info=exc,
                )
            try:
                await asyncio.sleep(backoff)
            except asyncio.CancelledError:
                raise
            backoff = min(backoff * 2.0, max_backoff)

    async def _acquire_plane_lock(self) -> None:
        self._plane_lock.acquire()
        logger.info(
            "Worker plane lock acquired",
            plane=self._plane_name,
            lock_path=str(self._plane_lock._path),
        )

    async def _release_plane_lock(self) -> None:
        try:
            self._plane_lock.release()
        except Exception as exc:
            logger.warning(
                "Worker plane lock release failed",
                plane=self._plane_name,
                lock_path=str(self._plane_lock._path),
                exc_info=exc,
            )

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

    def _record_crash_and_get_cooldown(self, module_name: str) -> tuple[int, float]:
        """Record this crash and return (crashes_in_last_hour, cooldown_seconds).

        Cooldown escalates so a thrashing worker doesn't hammer downstream
        services in tight restart loops:
          1-2 crashes/hr → 1s   (current behavior)
          3-5 crashes/hr → 5s
          6-9 crashes/hr → 15s
          10-19 crashes/hr → 60s + ERROR-level alert
          20-49 crashes/hr → 5 min
          50+ crashes/hr → 30 min

        SOAK-2026-05-18: the upper tiers were added after a 50-hour
        postgres outage produced 15+ crashes per worker in 5 minutes.
        The previous 60s ceiling meant each worker still relaunched every
        minute against an unreachable DB, generating 1000+ log lines and
        keeping the host CPU pegged on retry-spawn churn.  At 5-30 min
        cooldown the host effectively quiesces until the DB returns and
        recovers within the same hour-long crash window once the first
        crash-free cycle prunes the history.
        """
        import time as _time
        now = _time.monotonic()
        history = self._crash_history.setdefault(module_name, [])
        cutoff = now - 3600.0
        # Prune entries older than 1h
        while history and history[0] < cutoff:
            history.pop(0)
        history.append(now)
        crashes_in_window = len(history)
        if crashes_in_window >= 50:
            cooldown = 1800.0
        elif crashes_in_window >= 20:
            cooldown = 300.0
        elif crashes_in_window >= 10:
            cooldown = 60.0
        elif crashes_in_window >= 6:
            cooldown = 15.0
        elif crashes_in_window >= 3:
            cooldown = 5.0
        else:
            cooldown = 1.0
        return crashes_in_window, cooldown

    async def _restart_worker_task(self, module_name: str, *, reason: str) -> None:
        if self._shutting_down:
            return
        # Track crash + apply escalating cooldown BEFORE spawning the
        # replacement, so a thrashing worker has a backoff window between
        # failures rather than a flat 1s.
        crashes_in_window, cooldown_seconds = self._record_crash_and_get_cooldown(module_name)
        worker_label = _worker_name_from_module(module_name)
        if crashes_in_window >= 10:
            logger.error(
                "Worker thrashing — high crash rate",
                plane=self._plane_name,
                worker=worker_label,
                crashes_in_last_hour=crashes_in_window,
                cooldown_seconds=cooldown_seconds,
                reason=reason,
            )
        elif crashes_in_window >= 3:
            logger.warning(
                "Worker crash rate elevated",
                plane=self._plane_name,
                worker=worker_label,
                crashes_in_last_hour=crashes_in_window,
                cooldown_seconds=cooldown_seconds,
                reason=reason,
            )
        if cooldown_seconds > 1.0:
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=cooldown_seconds - 1.0)
                return  # shutdown requested during cooldown
            except asyncio.TimeoutError:
                pass

        current = self._worker_tasks.get(module_name)
        if current is not None:
            await self._cancel_worker_task(module_name, current)
        replacement = await self._spawn_worker_task(module_name)
        self._worker_tasks[module_name] = replacement
        # Record restart time so the freshness monitor gives the worker a
        # grace period to complete its first cycle and write a fresh
        # heartbeat, preventing an infinite restart loop.
        import time as _time
        self._restart_grace[module_name] = _time.monotonic()
        logger.warning(
            "Worker task restarted",
            plane=self._plane_name,
            worker=_worker_name_from_module(module_name),
            reason=reason,
            crashes_in_last_hour=crashes_in_window,
        )

    async def _monitor_worker_task(self, module_name: str) -> None:
        while not self._shutting_down:
            task = self._worker_tasks.get(module_name)
            if task is None:
                await asyncio.sleep(1.0)
                continue
            try:
                await task
                await asyncio.sleep(0)  # yield so parallel restart handlers update state first
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
                await asyncio.sleep(0)
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
                await asyncio.sleep(0)
                if self._shutting_down:
                    return
                if self._runtime_tasks.get(runtime_name) is not task:
                    continue
                logger.error("Runtime task exited unexpectedly", plane=self._plane_name, runtime=runtime_name)
                await asyncio.sleep(1.0)
                await self._restart_runtime_task(runtime_name, reason="unexpected_exit")
            except asyncio.CancelledError:
                await asyncio.sleep(0)
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
            # Split the two awaits so a recurring AttributeError /
            # type-coercion failure tells us *which* call raised it. The
            # combined try used to hide that distinction; the structured
            # logger's ``||`` summary line also drops the traceback
            # frames, so format and attach it explicitly.
            snapshots = None
            orchestrator_snapshot = None
            try:
                async with AsyncSessionLocal() as session:
                    snapshots = await list_worker_snapshots(session, include_stats=False)
            except Exception as exc:
                logger.warning(
                    "Worker freshness check failed (list_worker_snapshots)",
                    plane=self._plane_name,
                    exc_type=type(exc).__name__,
                    exc_msg=str(exc) or "<no message>",
                    traceback=traceback.format_exc(),
                    exc_info=exc,
                )
                continue
            try:
                async with AsyncSessionLocal() as session:
                    orchestrator_snapshot = await read_orchestrator_snapshot(session)
            except Exception as exc:
                logger.warning(
                    "Worker freshness check failed (read_orchestrator_snapshot)",
                    plane=self._plane_name,
                    exc_type=type(exc).__name__,
                    exc_msg=str(exc) or "<no message>",
                    traceback=traceback.format_exc(),
                    exc_info=exc,
                )
                continue

            snapshot_by_name = {
                str(item.get("worker_name") or ""): item for item in snapshots if isinstance(item, dict)
            }
            now = utcnow()

            import time as _time
            mono_now = _time.monotonic()

            for module_name, task in list(self._worker_tasks.items()):
                if task.done():
                    continue
                # Skip freshness check if this worker was recently restarted
                # and hasn't had enough time to complete its first cycle and
                # write a fresh heartbeat.  Without this, a worker with a
                # long interval (e.g. events at 300s) gets restarted every
                # 30s in an infinite loop because the old stale timestamp is
                # still in the DB.
                grace_until = self._restart_grace.get(module_name)
                if grace_until is not None:
                    grace_elapsed = mono_now - grace_until
                    worker_interval = max(1, int((snapshot_by_name.get(_worker_name_from_module(module_name)) or {}).get("interval_seconds") or 60))
                    # Give 2× the worker interval plus 60s buffer
                    grace_period = worker_interval * 2 + 60
                    if grace_elapsed < grace_period:
                        continue
                    else:
                        self._restart_grace.pop(module_name, None)
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

            orchestrator_runtime_snapshot = runtime_status.get_orchestrator("general")
            crypto_runtime_snapshot = runtime_status.get_orchestrator("crypto")
            if not isinstance(orchestrator_runtime_snapshot, dict) or not orchestrator_runtime_snapshot.get("updated_at"):
                orchestrator_runtime_snapshot = orchestrator_snapshot if isinstance(orchestrator_snapshot, dict) else None
            for runtime_name, task in list(self._runtime_tasks.items()):
                if task.done():
                    continue
                if runtime_name == "trader_orchestrator":
                    snapshot = orchestrator_runtime_snapshot
                    worker_name = "trader_orchestrator"
                else:
                    snapshot = crypto_runtime_snapshot
                    worker_name = "trader_orchestrator_crypto"
                if not snapshot:
                    continue
                updated_at = _parse_iso_utc(snapshot.get("updated_at"))
                if updated_at is None:
                    continue
                interval_seconds = max(1, int(snapshot.get("interval_seconds") or 5))
                stale_after_seconds = max(30, interval_seconds * 20) if worker_name == "trader_orchestrator_crypto" else max(60, interval_seconds * 12)
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

        # Bring up the Redis pool early so any subsequent service
        # initialization can opportunistically use the bus.  Soft-fail:
        # ``redis_client.start()`` returns False if Redis is disabled or
        # unreachable, and every consumer treats ``get_client_or_none()``
        # returning None as "use in-memory fallback".
        try:
            from services import redis_client

            redis_started = await redis_client.start()
            logger.info(
                "Redis client lifecycle started",
                plane=self._plane_name,
                healthy=redis_started,
                snapshot=redis_client.status_snapshot(),
            )
        except Exception as exc:
            logger.warning(
                "Redis client start failed (continuing with in-memory fallback)",
                plane=self._plane_name,
                exc_info=exc,
            )

        from models.database import start_pool_watchdog

        self._background_tasks.append(start_pool_watchdog())

        # Memory diagnostic loop — polls for the trigger file and
        # writes a snapshot when found.  Always running (cost ~ one
        # ``Path.exists()`` call every 30s) so an operator can dump
        # without restarting.  Tracemalloc is opt-in via env var
        # (see top of this file) — the loop still produces useful
        # psutil + module-singleton + GC sections without it.
        from utils.memory_diagnostic import memory_diagnostic_loop

        self._background_tasks.append(asyncio.create_task(
            memory_diagnostic_loop(),
            name="memory-diagnostic",
        ))

        # Hot-table pruner: keep ``trade_signals`` bounded so the fast
        # trader's ``list_unconsumed_trade_signals`` query stays under
        # the 2.5s fast-tier ``statement_timeout``.  Without this, the
        # table accretes 100K+ terminal rows in days and queries blow
        # past the timeout, corrupting asyncpg's protocol state with
        # cancelled-mid-flight queries (the "cannot switch to state X"
        # warnings).  Trading plane only — terminal signals must be
        # pruned where signals are produced + consumed.
        if self._plane_name == "trading":
            self._background_tasks.append(asyncio.create_task(
                self._run_trade_signal_pruner_loop(),
                name="trade-signal-pruner",
            ))

        # Hot-loop SLO alarm (Phase D): watch the trader orchestrator's
        # per-cycle lag and raise a throttled operator alert when it breaches
        # the configured SLA, so resource pressure that stalls trading surfaces
        # immediately instead of silently degrading fill latency. Trading plane
        # only — that's where the orchestrator runtime_status lives in-process.
        if self._plane_name == "trading":
            self._background_tasks.append(asyncio.create_task(
                self._run_orchestrator_slo_monitor_loop(),
                name="orchestrator-slo-monitor",
            ))

        # Stuck-skeleton retention sweep (plan 0011): DELETEs orphaned
        # ``trade_signals`` rows whose projection-loop UPSERT never
        # landed.  Lives on the discovery plane so the orphan-deletion
        # path stays off the trader-cycle's 10 s budget.  See
        # ``services.skeleton_signal_retention``.
        if self._plane_name == "discovery":
            self._background_tasks.append(asyncio.create_task(
                self._run_skeleton_signal_retention_loop(),
                name="skeleton-signal-retention",
            ))

        # Recording-session manager — promotes scheduled sessions, ticks
        # running ones, and auto-stops at scheduled_end_at /
        # max_duration_seconds.  Gated by ``start_recording`` so exactly ONE
        # plane (the dedicated ``recording`` plane) runs it — never two
        # managers racing the same sessions.
        if self._enabled("start_recording"):
            self._background_tasks.append(asyncio.create_task(
                self._run_recording_session_loop(),
                name="recording-session-manager",
            ))

        # Proactive recorder subscription — keeps the ISOLATED recording WS
        # pool subscribed to the top N liquid markets in MarketCatalog so
        # the microstructure recorder has book data for every market
        # a strategy might fire on.  Closes the BacktestStudio coverage
        # gap (38% no-data + 45% started-late on tail-end-carry's
        # 657 opp tokens).  Gated by ``start_recording`` — runs on the
        # dedicated ``recording`` plane, one pool + one subscription set,
        # off the trading event loop entirely.
        if self._enabled("start_recording"):
            self._background_tasks.append(asyncio.create_task(
                self._run_recorder_subscription_loop(),
                name="recorder-subscription",
            ))

        if self._enabled("load_strategy_registry"):
            try:
                from services.opportunity_strategy_catalog import ensure_all_strategies_seeded
                from services.strategy_loader import strategy_loader

                async with AsyncSessionLocal() as session:
                    seeded = await ensure_all_strategies_seeded(session)
                # Per-plane strategy filter.  Without this, every plane
                # imports every strategy at startup — the trading plane
                # ends up loading ``news_edge`` which transitively pulls
                # in torch + sentence_transformers (~2 GB heap).
                source_keys = self._plane_config.get("strategy_source_keys")
                if source_keys:
                    loaded = await strategy_loader.refresh_all_from_db(
                        source_keys=list(source_keys),
                        prune_unlisted=False,
                    )
                else:
                    loaded = await strategy_loader.refresh_all_from_db()
                logger.info(
                    "Strategy registries loaded",
                    plane=self._plane_name,
                    seeded=seeded.get("seeded", 0),
                    loaded=len(loaded.get("loaded", [])),
                    errors=len(loaded.get("errors", {})),
                    source_keys=list(source_keys) if source_keys else "all",
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
            intent_runtime = get_intent_runtime()
            self._schedule_background_startup(
                task_name="intent-runtime-init",
                starter=intent_runtime.start,
                failure_message="Intent runtime initialization failed",
                started_attr="_intent_runtime_started",
                started_check=lambda: intent_runtime.started,
            )

        if self._enabled("start_feed_manager"):
            feed_manager = get_feed_manager()
            self._schedule_background_startup(
                task_name="feed-manager-init",
                starter=feed_manager.start,
                failure_message="Feed manager initialization failed",
                started_attr="_feed_manager_started",
                started_check=lambda: bool(getattr(feed_manager, "_started", False)),
            )

        if self._enabled("start_market_runtime"):
            market_runtime = get_market_runtime()
            self._schedule_background_startup(
                task_name="market-runtime-init",
                starter=market_runtime.start,
                failure_message="Market runtime initialization failed",
                started_attr="_market_runtime_started",
                started_check=lambda: bool(getattr(market_runtime, "_started", False)),
            )

        if self._enabled("load_market_cache"):
            try:
                cache_load_task = market_cache_service.start_background_load()
                if cache_load_task is not None:
                    self._background_tasks.append(cache_load_task)
            except Exception as exc:
                logger.warning("Market cache load failed", plane=self._plane_name, exc_info=exc)

        if self._enabled("load_news_feed"):
            async def _load_news_feed() -> None:
                from services.news.feed_service import news_feed_service

                await news_feed_service.load_from_db()

            self._schedule_background_startup(
                task_name="news-feed-preload",
                starter=_load_news_feed,
                failure_message="News feed preload failed",
            )

        if self._enabled("initialize_live_execution"):
            self._background_tasks.append(
                asyncio.create_task(
                    self._initialize_live_execution_background(),
                    name=f"{self._plane_name}-live-execution-init",
                )
            )

        if self._enabled("start_copy_trade_service"):
            async def _start_copy_trade_service() -> None:
                await traders_copy_trade_signal_service.start()
                await traders_copy_trade_signal_service.refresh_scope()

            self._schedule_background_startup(
                task_name="copy-trade-service-init",
                starter=_start_copy_trade_service,
                failure_message="Copy-trade service initialization failed",
                started_attr="_copy_trade_service_started",
                started_check=lambda: bool(getattr(traders_copy_trade_signal_service, "_running", False)),
            )

        if self._enabled("start_position_monitor"):
            self._schedule_background_startup(
                task_name="position-monitor-init",
                starter=position_monitor.start,
                failure_message="Position monitor start failed",
                started_attr="_position_monitor_started",
                started_check=lambda: bool(getattr(position_monitor, "_running", False)),
            )

        if self._enabled("start_fill_monitor"):
            async def _start_fill_monitor() -> None:
                from services.fill_monitor import fill_monitor

                await fill_monitor.start()

            def _fill_monitor_started() -> bool:
                try:
                    from services.fill_monitor import fill_monitor
                except Exception:
                    return False
                return bool(getattr(fill_monitor, "_running", False))

            self._schedule_background_startup(
                task_name="fill-monitor-init",
                starter=_start_fill_monitor,
                failure_message="Fill monitor start failed",
                started_attr="_fill_monitor_started",
                started_check=_fill_monitor_started,
            )

        # Wallet-state heartbeat publisher: only the trading plane runs
        # the Polymarket user-channel WS, so only it has authoritative
        # wallet deltas.  Publish a periodic snapshot to Redis so the API
        # plane (and any future consumer) can surface cross-process
        # wallet freshness without running its own WS feed.
        if self._plane_name == "trading":
            try:
                from services import wallet_state_bus

                self._schedule_background_startup(
                    task_name="wallet-state-bus-publisher",
                    starter=wallet_state_bus.start_publisher,
                    failure_message="wallet_state_bus publisher start failed",
                )
            except Exception as exc:
                logger.warning(
                    "Failed to schedule wallet_state_bus publisher",
                    plane=self._plane_name,
                    exc_info=exc,
                )

        # Cross-plane trade-signal arrival bridge: subscribe to the Redis
        # signal channels and re-publish on the local event_bus so the
        # orchestrator + fast trader wake within ~1ms of a signal being
        # written by the news plane (instead of the 60s DB-poll fallback).
        if self._plane_name == "trading":
            try:
                from services import signal_bus_redis_bridge

                self._schedule_background_startup(
                    task_name="signal-bus-redis-bridge",
                    starter=signal_bus_redis_bridge.start,
                    failure_message="signal_bus_redis_bridge start failed",
                )
            except Exception as exc:
                logger.warning(
                    "Failed to schedule signal_bus_redis_bridge",
                    plane=self._plane_name,
                    exc_info=exc,
                )

        # R10-B.1e: TelegramNotifier moved OFF the trading plane so its
        # 5 s DB poll + 30+ s asyncpg holds can never steal budget from
        # the trader-orchestrator hot path.  The discovery plane has
        # spare cycles, owns the singleton (a second notifier would
        # double-deliver every message), and hosts the cross-plane
        # ``notifier_bridge`` subscriber so producers on other planes can
        # publish via Redis.
        if self._plane_name == "discovery":
            try:
                from services.notifier import notifier as notifier_service

                async def _start_notifier() -> None:
                    await notifier_service.start()

                self._schedule_background_startup(
                    task_name="notifier-start",
                    starter=_start_notifier,
                    failure_message="Autotrader notifier start failed (non-critical)",
                    started_attr="_notifier_started",
                    started_check=lambda: bool(getattr(notifier_service, "_started", False)),
                )

                from services import notifier_bridge

                self._schedule_background_startup(
                    task_name="notifier-bridge-subscriber",
                    starter=notifier_bridge.start_subscriber,
                    failure_message="notifier_bridge subscriber start failed",
                )
            except Exception as exc:
                logger.warning(
                    "Failed to schedule notifier / notifier_bridge",
                    plane=self._plane_name,
                    exc_info=exc,
                )

            # Signal cache: full-payload Redis subscriber that
            # eliminates the fast trader's per-cycle
            # ``list_unconsumed_trade_signals`` DB query.  Architecture:
            #   * One-shot bootstrap from DB at startup seeds the cache
            #     with all currently-pending signals.  Marks the cache
            #     ready — fast trader trusts empty results from here on.
            #   * Subscriber receives every subsequent signal payload
            #     via the ``signal_payloads`` Redis channel.
            #   * On Redis reconnect the subscriber re-bootstraps once
            #     to reconcile any publishes dropped during the gap.
            #   * Hot path: pure in-memory dict lookup, NO DB.
            try:
                from services import signal_cache

                async def _start_cache_pipeline() -> None:
                    # Bootstrap FIRST so the cache is hot before the
                    # subscriber starts forwarding live updates.  The
                    # subscriber also re-bootstraps on every connect,
                    # so this is "best-effort eager seed" — if it
                    # fails (DB unreachable), the subscriber-side
                    # bootstrap will catch up once the connection is
                    # established.
                    try:
                        upserted = await signal_cache.bootstrap_from_db()
                        logger.info(
                            "signal_cache eager bootstrap complete",
                            plane=self._plane_name,
                            upserted=upserted,
                        )
                    except Exception as exc:
                        logger.warning(
                            "signal_cache eager bootstrap failed; subscriber will retry on connect",
                            plane=self._plane_name,
                            exc_info=exc,
                        )
                    await signal_cache.start_subscriber()

                self._schedule_background_startup(
                    task_name="signal-cache-pipeline",
                    starter=_start_cache_pipeline,
                    failure_message="signal_cache pipeline start failed",
                )
            except Exception as exc:
                logger.warning(
                    "Failed to schedule signal_cache pipeline",
                    plane=self._plane_name,
                    exc_info=exc,
                )

            # Event-loop watchdog: detects asyncio stalls and dumps the
            # task list with stack frames so we can see exactly which
            # coroutine was monopolizing the loop.  Critical diagnostic
            # for "hot path is mysteriously slow" symptoms — sync code
            # on the loop or a non-yielding C-extension call shows up
            # in the dumped frame.  Trading plane only — that's where
            # the latency-sensitive work runs.
            try:
                from services import event_loop_watchdog

                self._schedule_background_startup(
                    task_name="event-loop-watchdog",
                    starter=event_loop_watchdog.start,
                    failure_message="event-loop watchdog start failed",
                )
            except Exception as exc:
                logger.warning(
                    "Failed to schedule event-loop watchdog",
                    plane=self._plane_name,
                    exc_info=exc,
                )

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
            # Pull task/future identity so "Future exception was never retrieved"
            # logs become actionable. Without this, the operator gets a stream
            # of "ConnectionError suppressed" with no clue which background
            # task lost its handle. Prefer 'task' if asyncio supplied it,
            # else 'future' if it's a Task instance.
            task_or_future = context.get("task") or context.get("future")
            task_name = None
            task_coro = None
            if task_or_future is not None:
                try:
                    task_name = task_or_future.get_name()
                except Exception:
                    task_name = None
                try:
                    coro = task_or_future.get_coro() if hasattr(task_or_future, "get_coro") else None
                    if coro is not None:
                        task_coro = getattr(coro, "__qualname__", None) or getattr(coro, "__name__", None)
                except Exception:
                    task_coro = None
            if _should_suppress_asyncio_exception(message, exc):
                logger.warning(
                    "Asyncio callback error (suppressed)",
                    plane=self._plane_name,
                    context_message=message,
                    error_type=type(exc).__name__ if exc is not None else None,
                    error=str(exc) if exc is not None else None,
                    task_name=task_name,
                    task_coro=task_coro,
                )
                return
            if exc is not None:
                logger.error(
                    "Unhandled asyncio exception",
                    plane=self._plane_name,
                    context_message=message,
                    task_name=task_name,
                    task_coro=task_coro,
                    exc_info=exc,
                )
            else:
                logger.error(
                    "Unhandled asyncio exception",
                    plane=self._plane_name,
                    context_message=message,
                    task_name=task_name,
                    task_coro=task_coro,
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

        background_tasks = list(self._background_tasks)
        for task in background_tasks:
            if not task.done():
                task.cancel()
        for task in background_tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
        self._background_tasks.clear()

        try:
            from services.fill_monitor import fill_monitor
        except Exception:
            fill_monitor = None

        if self._fill_monitor_started or (fill_monitor is not None and bool(getattr(fill_monitor, "_running", False))):
            try:
                if fill_monitor is not None:
                    fill_monitor.stop()
            except Exception:
                pass
            self._fill_monitor_started = False
        if self._position_monitor_started or bool(getattr(position_monitor, "_running", False)):
            position_monitor.stop()
            self._position_monitor_started = False
        if self._copy_trade_service_started or bool(getattr(traders_copy_trade_signal_service, "_running", False)):
            traders_copy_trade_signal_service.stop()
            self._copy_trade_service_started = False
        market_runtime = get_market_runtime()
        if self._market_runtime_started or bool(getattr(market_runtime, "_started", False)):
            await market_runtime.stop()
            self._market_runtime_started = False
        feed_manager = get_feed_manager()
        if self._feed_manager_started or bool(getattr(feed_manager, "_started", False)):
            await feed_manager.stop()
            self._feed_manager_started = False
        intent_runtime = get_intent_runtime()
        if self._intent_runtime_started or bool(getattr(intent_runtime, "started", False)):
            await intent_runtime.stop()
            self._intent_runtime_started = False
        if self._event_dispatcher_started:
            await event_dispatcher.stop()
            self._event_dispatcher_started = False
        if self._event_bus_started:
            await event_bus.stop()
            self._event_bus_started = False

        if self._plane_name == "trading":
            try:
                from services import wallet_state_bus

                await wallet_state_bus.stop_publisher()
            except Exception as exc:
                logger.warning(
                    "wallet_state_bus publisher stop raised",
                    plane=self._plane_name,
                    exc_info=exc,
                )
            try:
                from services import signal_bus_redis_bridge

                await signal_bus_redis_bridge.stop()
            except Exception as exc:
                logger.warning(
                    "signal_bus_redis_bridge stop raised",
                    plane=self._plane_name,
                    exc_info=exc,
                )

        if self._plane_name == "discovery":
            # R10-B.1e: stop notifier bridge subscriber first (no more
            # incoming messages), then stop the notifier itself.
            try:
                from services import notifier_bridge

                await notifier_bridge.stop_subscriber()
            except Exception as exc:
                logger.warning(
                    "notifier_bridge stop raised",
                    plane=self._plane_name,
                    exc_info=exc,
                )
            try:
                from services.notifier import notifier as notifier_service

                await notifier_service.shutdown()
            except Exception as exc:
                logger.warning(
                    "Notifier shutdown raised",
                    plane=self._plane_name,
                    exc_info=exc,
                )

        if self._plane_name == "trading":
            try:
                from services import signal_cache

                await signal_cache.stop_subscriber()
            except Exception as exc:
                logger.warning(
                    "signal_cache subscriber stop raised",
                    plane=self._plane_name,
                    exc_info=exc,
                )
            try:
                from services import event_loop_watchdog

                await event_loop_watchdog.stop()
            except Exception as exc:
                logger.warning(
                    "event-loop watchdog stop raised",
                    plane=self._plane_name,
                    exc_info=exc,
                )

        try:
            from services import redis_client

            await redis_client.shutdown()
        except Exception as exc:
            logger.warning(
                "Redis client shutdown raised",
                plane=self._plane_name,
                exc_info=exc,
            )

        if self._cpu_executor is not None:
            self._cpu_executor.shutdown(wait=False, cancel_futures=True)
            self._cpu_executor = None

    async def run(self) -> None:
        await self._acquire_plane_lock()
        await self._start_plane()
        loop = asyncio.get_running_loop()

        def _request_stop() -> None:
            self._stop_event.set()

        _signal_handlers_installed = False
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _request_stop)
                _signal_handlers_installed = True
            except (NotImplementedError, RuntimeError, ValueError):
                pass

        if not _signal_handlers_installed and os.name == "nt":
            # Windows: loop.add_signal_handler is not supported.
            # Use signal.signal for SIGINT (Ctrl+C) and call_soon_threadsafe
            # to set the stop event from the signal handler thread.
            def _win_sigint_handler(signum: int, frame: Any) -> None:
                try:
                    loop.call_soon_threadsafe(_request_stop)
                except RuntimeError:
                    pass  # loop already closed

            signal.signal(signal.SIGINT, _win_sigint_handler)

        await self._stop_event.wait()

    async def shutdown(self) -> None:
        try:
            await self._stop_plane()
        finally:
            await self._release_plane_lock()


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


def _select_fast_event_loop_policy() -> None:
    """Pick the fastest available asyncio event loop for the worker plane.

    Workers never spawn subprocesses (asyncio.create_subprocess_* only appears
    in services/validation_service.py, which runs in the API process), so we
    are not bound by the stdlib ProactorEventLoop requirement and can pick
    whichever loop is fastest for pure socket I/O.

    Cross-platform preference order:
      * Linux + macOS: uvloop (libuv-based; the de-facto standard for
        high-throughput asyncio).  Skipped on Windows because uvloop is
        POSIX-only.
      * Windows: winloop (uvloop's Windows port, IOCP-based).  Falls back
        to WindowsSelectorEventLoopPolicy on ImportError.  We intentionally
        avoid stdlib WindowsProactorEventLoopPolicy because of historical
        _ProactorReadPipeTransport._loop_reading crashes under load —
        winloop is a separate IOCP implementation that does not share that
        code path.

    Why not also override the API process?  validation_service.py uses
    asyncio.create_subprocess_exec which requires Proactor on Windows.
    The API entry point therefore stays on Python defaults.

    Opt out via HOMERUN_FAST_LOOP=0 (also accepts ``false``/``no``/``off``).
    Backwards compatible with the prior HOMERUN_USE_WINLOOP=0 toggle.

    2026-05-20 (Windows): measured 30% lower asyncpg per-query latency at
    8-40 concurrent connections vs WindowsSelectorEventLoopPolicy.  Linux +
    macOS gains are typically larger because uvloop is more battle-tested.
    """
    opt_out = (
        os.environ.get("HOMERUN_FAST_LOOP", "").strip().lower() in {"0", "false", "no", "off"}
        or os.environ.get("HOMERUN_USE_WINLOOP", "").strip().lower() in {"0", "false", "no", "off"}
    )
    if opt_out:
        if os.name == "nt":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        return

    if os.name != "nt":
        try:
            import uvloop  # type: ignore[import-not-found]

            asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
            return
        except ImportError:
            # No uvloop available — leave Python's default (epoll/kqueue) in place.
            return

    # Windows
    try:
        import winloop  # type: ignore[import-not-found]

        asyncio.set_event_loop_policy(winloop.EventLoopPolicy())
    except ImportError:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def main() -> None:
    _select_fast_event_loop_policy()
    args = _parse_args()
    plane_name = str(args.plane)
    # Plane env var so plane-specific gating (e.g. user-channel WS
    # feed only on ``trading``) can detect the plane without passing
    # arguments through every callsite.
    os.environ["HOMERUN_WORKER_PLANE"] = plane_name
    asyncio.run(_run(plane_name))


if __name__ == "__main__":
    main()
