import os
import asyncio
import signal
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from utils.utcnow import utcnow
from fastapi import FastAPI, WebSocket, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pathlib import Path
from typing import Optional
from sqlalchemy import select

# Keep native ML/linear algebra threading conservative for long-running
# backend and worker workloads on macOS.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("NEWS_FAISS_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("EMBEDDING_DEVICE", "cpu")

from config import settings, RUNTIME_SETTINGS_PRECEDENCE
from api import router, handle_websocket
from api.routes_simulation import simulation_router
from api.routes_anomaly import anomaly_router
from api.routes_orchestrator_live import router as orchestrator_live_router
from api.routes_maintenance import router as maintenance_router
from api.routes_settings import router as settings_router
from api.routes_ui_lock import router as ui_lock_router
from api.routes_ai import router as ai_router
from api.routes_news import router as news_router
from api.routes_discovery import discovery_router
from api.routes_kalshi import router as kalshi_router
from api.routes_crypto import router as crypto_router
from api.routes_news_workflow import router as news_workflow_router
from api.routes_weather_workflow import router as weather_workflow_router
from api.routes_events import router as events_router
from api.routes_signals import router as signals_router
from api.routes_workers import router as workers_router
from api.routes_validation import router as validation_router
from api.routes_trader_orchestrator import router as trader_orchestrator_router
from api.routes_trader_sources import router as trader_sources_router
from api.routes_strategies import router as strategies_router
from api.routes_ml import router as ml_router
from api.routes_data_sources import router as data_sources_router
from api.routes_traders import router as traders_router
from services.wallet_tracker import wallet_tracker
from services.live_execution_service import live_execution_service
from services.wallet_discovery import wallet_discovery
from services.maintenance import maintenance_service
from services.validation_service import validation_service
from services.snapshot_broadcaster import snapshot_broadcaster
from services.market_prioritizer import market_prioritizer
from services.event_bus import event_bus
from services.event_dispatcher import event_dispatcher
from models.database import AppSettings, AsyncSessionLocal, init_database
from models.model_registry import register_all_models
from services import discovery_shared_state, shared_state
from services.news import shared_state as news_shared_state
from services.pause_state import global_pause_state
from services.trader_orchestrator_state import (
    enforce_manual_start_on_startup,
    read_orchestrator_control,
    read_orchestrator_snapshot,
)
from services.weather import shared_state as weather_shared_state
from services.worker_state import list_worker_snapshots, read_worker_control, summarize_worker_stats
from services.ui_lock import UI_LOCK_SESSION_COOKIE, ui_lock_service
from utils.logger import setup_logging, get_logger
from utils.rate_limiter import rate_limiter, TokenBucket

register_all_models()

# Setup logging
setup_logging(level=settings.LOG_LEVEL if hasattr(settings, "LOG_LEVEL") else "INFO")
logger = get_logger("main")


class InboundAPIRateLimiter:
    def __init__(self) -> None:
        self._buckets: dict[str, TokenBucket] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._last_seen: dict[str, float] = {}
        self._last_cleanup: float = 0.0

    def _lock_for(self, key: str) -> asyncio.Lock:
        lock = self._locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[key] = lock
        return lock

    def _bucket_for(self, key: str) -> TokenBucket:
        bucket = self._buckets.get(key)
        if bucket is not None:
            return bucket

        capacity = max(1, int(settings.API_RATE_LIMIT_BURST))
        window_seconds = max(1, int(settings.API_RATE_LIMIT_WINDOW_SECONDS))
        refill_rate = max(1, int(settings.API_RATE_LIMIT_REQUESTS_PER_WINDOW)) / float(window_seconds)
        bucket = TokenBucket(
            capacity=float(capacity),
            tokens=float(capacity),
            refill_rate=float(refill_rate),
        )
        self._buckets[key] = bucket
        return bucket

    def _cleanup(self, now_monotonic: float) -> None:
        if now_monotonic - self._last_cleanup < 60:
            return
        stale_after_seconds = max(300, int(settings.API_RATE_LIMIT_WINDOW_SECONDS) * 20)
        stale_keys = [key for key, ts in self._last_seen.items() if now_monotonic - ts >= stale_after_seconds]
        for key in stale_keys:
            self._last_seen.pop(key, None)
            self._buckets.pop(key, None)
            self._locks.pop(key, None)
        self._last_cleanup = now_monotonic

    async def consume(self, client_key: str) -> tuple[bool, float, float]:
        key = str(client_key or "unknown")
        lock = self._lock_for(key)
        async with lock:
            bucket = self._bucket_for(key)
            bucket.refill()
            now_monotonic = asyncio.get_running_loop().time()
            self._last_seen[key] = now_monotonic
            self._cleanup(now_monotonic)
            if bucket.tokens >= 1:
                bucket.consume(1)
                return True, 0.0, bucket.tokens
            wait_seconds = bucket.wait_time(1)
            return False, wait_seconds, bucket.tokens

    def status(self) -> dict[str, object]:
        return {
            "enabled": bool(settings.API_RATE_LIMIT_ENABLED),
            "requests_per_window": int(settings.API_RATE_LIMIT_REQUESTS_PER_WINDOW),
            "window_seconds": int(settings.API_RATE_LIMIT_WINDOW_SECONDS),
            "burst": int(settings.API_RATE_LIMIT_BURST),
            "tracked_clients": len(self._buckets),
        }


inbound_api_rate_limiter = InboundAPIRateLimiter()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    logger.info("Starting Autonomous Prediction Market Trading Platform...")

    # Create a shared thread pool executor for CPU-bound work so that
    # heavy background tasks (strategy detection, ML embedding, wallet
    # analysis) never block the async event loop that serves API requests.
    cpu_count = os.cpu_count() or 4
    cpu_executor = ThreadPoolExecutor(
        max_workers=max(cpu_count * 2 + 8, 16),
        thread_name_prefix="cpu-pool",
    )
    loop = asyncio.get_running_loop()
    loop.set_default_executor(cpu_executor)

    def _global_exception_handler(loop_ref, context):
        exc = context.get("exception")
        message = context.get("message", "Unhandled asyncio exception")
        if exc is not None:
            logger.error("Unhandled asyncio exception: %s", message, exc_info=exc)
        else:
            logger.error("Unhandled asyncio exception: %s", message)

    loop.set_exception_handler(_global_exception_handler)

    logger.info(
        "Thread pool executor configured",
        max_workers=max(cpu_count * 2 + 8, 16),
    )
    tasks: list[asyncio.Task] = []

    try:
        # Initialize database
        await init_database()
        logger.info("Database initialized")

        # Start pool watchdog — reaps connections held too long and
        # recovers the pool when near-exhaustion is detected.
        from models.database import start_pool_watchdog
        pool_watchdog_task = start_pool_watchdog()
        tasks.append(pool_watchdog_task)

        # Warm unified strategy loader at process startup.
        try:
            from services.opportunity_strategy_catalog import ensure_all_strategies_seeded
            from services.strategy_loader import strategy_loader as _loader

            async with AsyncSessionLocal() as session:
                seeded = await ensure_all_strategies_seeded(session)
                loaded = await _loader.refresh_all_from_db(session=session)
            logger.info(
                "Strategy registries loaded",
                seeded=seeded.get("seeded", 0),
                loaded=len(loaded.get("loaded", [])),
                errors=len(loaded.get("errors", {})),
            )
        except Exception as e:
            logger.warning(f"Failed to preload strategy registries: {e}")

        # Warm unified data source loader at process startup.
        try:
            from services.data_source_catalog import ensure_all_data_sources_seeded
            from services.data_source_loader import data_source_loader

            async with AsyncSessionLocal() as session:
                seeded = await ensure_all_data_sources_seeded(session)
                loaded = await data_source_loader.refresh_all_from_db(session=session)
            logger.info(
                "Data source registries loaded",
                seeded=seeded.get("seeded", 0),
                loaded=len(loaded.get("loaded", [])),
                errors=len(loaded.get("errors", {})),
            )
        except Exception as e:
            logger.warning(f"Failed to preload data source registries: {e}")

        await event_bus.start()
        await event_dispatcher.start()

        # Apply all DB runtime overrides using one deterministic precedence chain.
        try:
            from config import apply_runtime_settings_overrides

            await apply_runtime_settings_overrides()
            logger.info(
                "Runtime settings overrides applied",
                precedence=RUNTIME_SETTINGS_PRECEDENCE,
            )
        except Exception as exc:
            logger.warning(
                "Failed to apply runtime settings overrides (using env/defaults)",
                precedence=RUNTIME_SETTINGS_PRECEDENCE,
                exc_info=exc,
            )

        try:
            async with AsyncSessionLocal() as session:
                await enforce_manual_start_on_startup(session)
            logger.info("Trader orchestrator reset to manual-start mode at startup")
        except Exception as exc:
            logger.warning(
                "Failed to reset trader orchestrator to manual-start mode at startup",
                exc_info=exc,
            )

        # Restore global pause state from persisted worker controls.
        # This keeps API-owned loops (wallet tracker, LLM/trading gates)
        # aligned with worker controls across restarts.
        try:
            async with AsyncSessionLocal() as session:
                scanner_control = await shared_state.read_scanner_control(session)
                news_control = await news_shared_state.read_news_control(session)
                weather_control = await weather_shared_state.read_weather_control(session)
                discovery_control = await discovery_shared_state.read_discovery_control(session)
                orchestrator_control = await read_orchestrator_control(session)
                scanner_slo_control = await read_worker_control(session, "scanner_slo")
                crypto_control = await read_worker_control(session, "crypto")
                tracked_control = await read_worker_control(session, "tracked_traders")
                events_control = await read_worker_control(session, "events")

            should_pause = all(
                bool(control.get("is_paused", False))
                for control in (
                    scanner_control,
                    news_control,
                    weather_control,
                    discovery_control,
                    orchestrator_control,
                    scanner_slo_control,
                    crypto_control,
                    tracked_control,
                    events_control,
                )
            )
            if should_pause:
                global_pause_state.pause()
            else:
                global_pause_state.resume()
            logger.info("Global pause state restored", paused=should_pause)
        except Exception as e:
            logger.warning(f"Failed to restore global pause state (continuing): {e}")

        # Pre-flight configuration validation
        from services.config_validator import config_validator

        validation = config_validator.validate_all(settings)
        if not validation.valid:
            logger.error(
                "Configuration validation failed",
                errors=validation.errors,
                warnings=validation.warnings,
            )
        elif validation.warnings:
            logger.warning(f"Config warnings: {validation.warnings}")

        # Load sport token classifications from DB
        try:
            from services.sport_classifier import sport_classifier

            await sport_classifier.load_from_db()
        except Exception as e:
            logger.warning(f"Sport classifier load failed (non-critical): {e}")

        # Load persistent market cache from DB into memory
        try:
            from services.market_cache import market_cache_service

            await market_cache_service.load_from_db()
            stats = await market_cache_service.get_cache_stats()
            logger.info(
                "Market cache loaded from DB",
                markets=stats.get("market_count", 0),
                usernames=stats.get("username_count", 0),
            )
        except Exception as e:
            logger.warning(f"Market cache load failed (non-critical): {e}")

        # Initialize AI intelligence layer
        try:
            from services.ai import initialize_ai
            from services.ai.skills.loader import skill_loader

            llm_manager = await initialize_ai()
            skill_loader.discover()
            if llm_manager.is_available():
                logger.info(
                    "AI intelligence layer initialized",
                    providers=list(llm_manager._providers.keys()),
                    skills=len(skill_loader.list_skills()),
                )
            else:
                logger.info("AI intelligence layer initialized (no providers configured)")
        except Exception as e:
            logger.warning(f"AI initialization failed (non-critical): {e}")

        # Add any preconfigured wallets
        for wallet in settings.TRACKED_WALLETS:
            await wallet_tracker.add_wallet(wallet)

        # API-owned background tasks. The worker plane runs in the dedicated
        # workers.host process and writes durable snapshots back to the DB.

        # Broadcast scanner snapshot deltas from DB to connected WebSocket clients.
        await snapshot_broadcaster.start(interval_seconds=1.0)

        wallet_task = asyncio.create_task(wallet_tracker.start_monitoring(30))
        tasks.append(wallet_task)

        # Initialize live execution service if credentials are configured
        trading_initialized = await live_execution_service.initialize()
        if trading_initialized:
            logger.info("Live execution service initialized")
        else:
            logger.info("Live execution service not initialized - credentials not configured")

        # Start background cleanup if enabled (DB settings override env defaults).
        cleanup_enabled = bool(settings.AUTO_CLEANUP_ENABLED)
        cleanup_interval_hours = int(settings.CLEANUP_INTERVAL_HOURS)
        cleanup_config = {
            "resolved_trade_days": int(settings.CLEANUP_RESOLVED_TRADE_DAYS),
            "open_trade_expiry_days": int(settings.CLEANUP_OPEN_TRADE_EXPIRY_DAYS),
            "wallet_trade_days": int(settings.CLEANUP_WALLET_TRADE_DAYS),
            "anomaly_days": int(settings.CLEANUP_ANOMALY_DAYS),
            "trade_signal_emission_days": int(settings.CLEANUP_TRADE_SIGNAL_EMISSION_DAYS),
            "trade_signal_update_days": int(settings.CLEANUP_TRADE_SIGNAL_UPDATE_DAYS),
            "wallet_activity_rollup_days": int(settings.CLEANUP_WALLET_ACTIVITY_ROLLUP_DAYS),
            "wallet_activity_dedupe_enabled": bool(settings.CLEANUP_WALLET_ACTIVITY_DEDUPE_ENABLED),
        }
        try:
            async with AsyncSessionLocal() as session:
                row = (
                    await session.execute(select(AppSettings).where(AppSettings.id == "default"))
                ).scalar_one_or_none()
                if row is not None:
                    if row.auto_cleanup_enabled is not None:
                        cleanup_enabled = bool(row.auto_cleanup_enabled)
                    cleanup_interval_hours = int(row.cleanup_interval_hours or cleanup_interval_hours)
                    cleanup_config["resolved_trade_days"] = int(
                        row.cleanup_resolved_trade_days or cleanup_config["resolved_trade_days"]
                    )
                    cleanup_config["trade_signal_emission_days"] = int(
                        row.cleanup_trade_signal_emission_days or cleanup_config["trade_signal_emission_days"]
                    )
                    cleanup_config["trade_signal_update_days"] = int(
                        row.cleanup_trade_signal_update_days
                        if row.cleanup_trade_signal_update_days is not None
                        else cleanup_config["trade_signal_update_days"]
                    )
                    cleanup_config["wallet_activity_rollup_days"] = int(
                        row.cleanup_wallet_activity_rollup_days or cleanup_config["wallet_activity_rollup_days"]
                    )
                    cleanup_config["wallet_activity_dedupe_enabled"] = bool(
                        row.cleanup_wallet_activity_dedupe_enabled
                        if row.cleanup_wallet_activity_dedupe_enabled is not None
                        else cleanup_config["wallet_activity_dedupe_enabled"]
                    )
        except Exception as e:
            logger.warning("Failed to load DB maintenance settings; using env defaults", exc_info=e)

        if cleanup_enabled:
            cleanup_task = asyncio.create_task(
                maintenance_service.start_background_cleanup(
                    interval_hours=cleanup_interval_hours,
                    cleanup_config=cleanup_config,
                )
            )
            tasks.append(cleanup_task)
            logger.info(
                "Background database cleanup enabled",
                interval_hours=cleanup_interval_hours,
            )

        # Initialize news intelligence layer (workers own background loops)
        try:
            from services.news.feed_service import news_feed_service
            from services.news.semantic_matcher import semantic_matcher

            # Load previously-cached articles from DB so they're available
            # immediately for matching and search.
            await news_feed_service.load_from_db()

            matcher_ready = False
            try:
                matcher_ready = await asyncio.wait_for(
                    asyncio.to_thread(semantic_matcher.initialize),
                    timeout=5.0,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Semantic matcher initialization timed out; continuing in deferred mode",
                    timeout_seconds=5.0,
                )
            logger.info(
                "News intelligence layer initialized (worker-owned execution)",
                ml_mode=bool(semantic_matcher.is_ml_mode and matcher_ready),
                deferred_init=not matcher_ready,
                cached_articles=news_feed_service.article_count,
            )
        except Exception as e:
            logger.warning(f"News intelligence init failed (non-critical): {e}")

        # Notifier and opportunity_recorder run in scanner worker (callbacks on scan)

        # Start validation orchestration (async job queue + guardrails)
        await validation_service.start()

        logger.info("All services started successfully")

        yield

    except Exception as e:
        logger.critical("Startup failed", exc_info=e)
        raise

    finally:
        # Cleanup
        logger.info("Shutting down...")

        await snapshot_broadcaster.stop()
        wallet_tracker.stop()
        wallet_discovery.stop()
        maintenance_service.stop()
        await validation_service.stop()
        try:
            from services.news.feed_service import news_feed_service

            news_feed_service.stop()
        except Exception:
            pass

        for task in tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        await event_dispatcher.stop()
        await event_bus.stop()

        try:
            from services.polymarket import polymarket_client

            await polymarket_client.close()
        except Exception:
            pass
        cpu_executor.shutdown(wait=False)
        logger.info("Shutdown complete")


app = FastAPI(
    title="Homerun",
    description="Polymarket arbitrage detection, paper trading, and autonomous trading",
    version="2.0.0",
    lifespan=lifespan,
)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(
        "Unhandled exception",
        path=request.url.path,
        method=request.method,
        exc_info=exc,
    )
    return JSONResponse(status_code=500, content={"detail": "Internal server error", "error": str(exc)})


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS if hasattr(settings, "CORS_ORIGINS") else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Total-Count"],
)


_UI_LOCK_EXEMPT_API_PATHS = {
    "/api/ui-lock/status",
    "/api/ui-lock/unlock",
    "/api/ui-lock/lock",
    "/api/ui-lock/activity",
}


@app.middleware("http")
async def ui_lock_guard(request: Request, call_next):
    path = request.url.path
    if not path.startswith("/api"):
        return await call_next(request)
    if path in _UI_LOCK_EXEMPT_API_PATHS:
        return await call_next(request)

    token = request.cookies.get(UI_LOCK_SESSION_COOKIE)
    unlocked = await ui_lock_service.is_token_unlocked(token)
    if not unlocked:
        status = await ui_lock_service.status(token)
        if status.get("enabled"):
            return JSONResponse(
                status_code=423,
                content={
                    "detail": "UI lock is active.",
                    "code": "ui_locked",
                    "ui_lock": status,
                },
            )
    return await call_next(request)


@app.middleware("http")
async def inbound_api_rate_limit(request: Request, call_next):
    if not bool(settings.API_RATE_LIMIT_ENABLED):
        return await call_next(request)
    if request.method.upper() == "OPTIONS":
        return await call_next(request)
    if not request.url.path.startswith("/api"):
        return await call_next(request)

    client_ip = request.headers.get("x-forwarded-for", "")
    if client_ip:
        client_ip = client_ip.split(",")[0].strip()
    if not client_ip:
        client_ip = request.client.host if request.client else "unknown"
    route_key = request.url.path
    client_key = f"{client_ip}:{route_key}"

    allowed, wait_seconds, remaining_tokens = await inbound_api_rate_limiter.consume(client_key)
    if not allowed:
        retry_after = max(1, int(wait_seconds + 0.999))
        return JSONResponse(
            status_code=429,
            content={
                "detail": "API rate limit exceeded. Please retry shortly.",
                "retry_after_seconds": retry_after,
            },
            headers={
                "Retry-After": str(retry_after),
                "X-RateLimit-Limit": str(int(settings.API_RATE_LIMIT_REQUESTS_PER_WINDOW)),
                "X-RateLimit-Window": str(int(settings.API_RATE_LIMIT_WINDOW_SECONDS)),
                "X-RateLimit-Remaining": str(int(max(0.0, remaining_tokens))),
            },
        )

    response = await call_next(request)
    response.headers["X-RateLimit-Limit"] = str(int(settings.API_RATE_LIMIT_REQUESTS_PER_WINDOW))
    response.headers["X-RateLimit-Window"] = str(int(settings.API_RATE_LIMIT_WINDOW_SECONDS))
    response.headers["X-RateLimit-Remaining"] = str(int(max(0.0, remaining_tokens)))
    return response


# API routes
app.include_router(router, prefix="/api")
app.include_router(simulation_router, prefix="/api/simulation", tags=["Simulation"])
app.include_router(anomaly_router, prefix="/api/anomaly", tags=["Anomaly Detection"])
app.include_router(orchestrator_live_router, prefix="/api", tags=["Trader Orchestrator"])
app.include_router(trader_orchestrator_router, prefix="/api", tags=["Trader Orchestrator"])
app.include_router(traders_router, prefix="/api", tags=["Traders"])
app.include_router(trader_sources_router, prefix="/api", tags=["Trader Sources"])
app.include_router(maintenance_router, prefix="/api", tags=["Maintenance"])
app.include_router(settings_router, prefix="/api", tags=["Settings"])
app.include_router(ui_lock_router, prefix="/api", tags=["UI Lock"])
app.include_router(ai_router, prefix="/api", tags=["AI Intelligence"])
app.include_router(news_router, prefix="/api", tags=["News Intelligence"])
app.include_router(discovery_router, prefix="/api/discovery", tags=["Trader Discovery"])
app.include_router(kalshi_router, prefix="/api", tags=["Kalshi"])
# Unified strategies router at /api/strategies/* (registered after legacy routers)
app.include_router(strategies_router, prefix="/api", tags=["Strategies (Unified)"])
app.include_router(data_sources_router, prefix="/api", tags=["Data Sources"])
app.include_router(crypto_router, prefix="/api", tags=["Crypto Markets"])
app.include_router(news_workflow_router, prefix="/api", tags=["News Workflow"])
app.include_router(weather_workflow_router, prefix="/api", tags=["Weather Workflow"])
app.include_router(signals_router, prefix="/api", tags=["Signals"])
app.include_router(workers_router, prefix="/api", tags=["Workers"])
app.include_router(validation_router, prefix="/api", tags=["Validation"])
app.include_router(events_router, prefix="/api", tags=["Events"])
app.include_router(ml_router, prefix="/api", tags=["ML Training Pipeline"])


# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    token = websocket.cookies.get(UI_LOCK_SESSION_COOKIE)
    unlocked = await ui_lock_service.is_token_unlocked(token)
    if not unlocked:
        status = await ui_lock_service.status(token)
        if status.get("enabled"):
            await websocket.accept()
            await websocket.send_json(
                {
                    "type": "ui_locked",
                    "data": {"message": "UI lock is active.", "ui_lock": status},
                }
            )
            await websocket.close(code=4403)
            return
    await handle_websocket(websocket)


# Health checks
@app.get("/health")
async def health_check():
    """Basic health check - for load balancers"""
    return {"status": "ok"}


@app.get("/health/live")
async def liveness_check():
    """Liveness probe - is the service running?"""
    return {"status": "alive", "timestamp": utcnow().isoformat()}


@app.get("/health/ready")
async def readiness_check():
    """Readiness probe - is the service ready to accept traffic?"""
    async with AsyncSessionLocal() as session:
        scanner_status = await shared_state.get_scanner_status_from_db(session)
    checks = {
        "scanner": scanner_status.get("running", False),
        "database": True,
        "ws_feeds": bool(_get_ws_feeds_status(scanner_status).get("healthy", False)) if settings.WS_FEED_ENABLED else True,
        "polymarket_api": True,
    }

    all_ready = all(checks.values())

    return {
        "status": "ready" if all_ready else "not_ready",
        "checks": checks,
        "api_rate_limit": inbound_api_rate_limiter.status(),
        "timestamp": utcnow().isoformat(),
    }


_tui_health_cache: Optional[dict] = None
_tui_health_refresh_task: Optional[asyncio.Task] = None
_tui_health_cache_updated_at: datetime | None = None
_TUI_HEALTH_CACHE_TTL_SECONDS = 5.0
_TUI_HEALTH_WORKER_NAMES = (
    "scanner",
    "scanner_slo",
    "discovery",
    "weather",
    "news",
    "crypto",
    "tracked_traders",
    "trader_orchestrator",
    "trader_reconciliation",
    "redeemer",
    "events",
)


async def _tui_health_db_queries() -> dict:
    """DB-dependent portion of the TUI health check."""
    async with AsyncSessionLocal() as session:
        scanner_status = await shared_state.get_scanner_status_from_db(session)
        worker_status_rows = await list_worker_snapshots(
            session,
            include_stats=False,
            worker_names=_TUI_HEALTH_WORKER_NAMES,
        )
        orchestrator_snapshot = await read_orchestrator_snapshot(session)
    return {
        "scanner_status": scanner_status,
        "worker_status_rows": worker_status_rows,
        "orchestrator_snapshot": orchestrator_snapshot,
        "database": True,
    }


def _build_tui_health_response(db: dict) -> dict:
    scanner_status = db.get("scanner_status", {})
    worker_status_rows = db.get("worker_status_rows", [])
    orchestrator_snapshot = db.get("orchestrator_snapshot", {})
    worker_status = {
        str(row.get("worker_name") or ""): {
            "worker_name": row.get("worker_name"),
            "running": bool(row.get("running", False)),
            "enabled": bool(row.get("enabled", False)),
            "current_activity": row.get("current_activity"),
            "interval_seconds": row.get("interval_seconds"),
            "last_run_at": row.get("last_run_at"),
            "lag_seconds": row.get("lag_seconds"),
            "last_error": row.get("last_error"),
            "updated_at": row.get("updated_at"),
        }
        for row in worker_status_rows
        if isinstance(row, dict) and str(row.get("worker_name") or "")
    }
    news_worker = worker_status.get("news", {})
    discovery_worker = worker_status.get("discovery", {})
    return {
        "status": "healthy",
        "timestamp": utcnow().isoformat(),
        "workers": worker_status,
        "checks": {
            "database": db.get("database", False),
        },
        "services": {
            "scanner": {
                "running": scanner_status.get("running", False),
                "last_scan": scanner_status.get("last_scan"),
                "opportunities_count": scanner_status.get("opportunities_count", 0),
            },
            "trader_orchestrator": {
                "running": bool(orchestrator_snapshot.get("running", False)),
                "current_activity": orchestrator_snapshot.get("current_activity"),
                "last_run_at": orchestrator_snapshot.get("last_run_at"),
                "last_error": orchestrator_snapshot.get("last_error"),
            },
            "ws_feeds": _get_ws_feeds_status(scanner_status, worker_status),
            "signal_runtime": {
                "queue_depth": _get_signal_runtime_status(),
            },
            "news_workflow": {
                "running": bool(news_worker.get("running", False)),
                "enabled": bool(news_worker.get("enabled", False)),
                "paused": False,
                "last_scan": news_worker.get("last_run_at"),
                "next_scan": None,
                "current_activity": news_worker.get("current_activity"),
                "last_error": news_worker.get("last_error"),
                "degraded_mode": False,
                "pending_intents": 0,
            },
            "wallet_discovery": {
                "running": bool(discovery_worker.get("running", False)),
                "last_run": discovery_worker.get("last_run_at"),
                "wallets_discovered": 0,
                "wallets_analyzed": 0,
                "current_activity": discovery_worker.get("current_activity"),
                "interval_minutes": None,
                "paused": False,
            },
            "api_rate_limit": inbound_api_rate_limiter.status(),
        },
    }


@app.get("/health/tui")
async def tui_health_check():
    """Lightweight health snapshot for high-frequency TUI polling.

    The DB queries are wrapped in a short timeout so that pool contention
    doesn't block the response and cause the TUI to flap OFFLINE/ONLINE.
    When the pool is busy the last successful result is served instead.
    """
    global _tui_health_cache, _tui_health_cache_updated_at, _tui_health_refresh_task
    if _tui_health_refresh_task is not None and _tui_health_refresh_task.done():
        try:
            _tui_health_cache = _tui_health_refresh_task.result()
            _tui_health_cache_updated_at = utcnow()
        except Exception:
            pass
        finally:
            _tui_health_refresh_task = None
    cache_age_seconds = None
    if _tui_health_cache_updated_at is not None:
        cache_age_seconds = max(0.0, (utcnow() - _tui_health_cache_updated_at).total_seconds())
    cache_is_fresh = bool(
        _tui_health_cache is not None
        and cache_age_seconds is not None
        and cache_age_seconds <= _TUI_HEALTH_CACHE_TTL_SECONDS
    )
    if cache_is_fresh:
        return _build_tui_health_response(_tui_health_cache or {"database": False})
    if _tui_health_refresh_task is None:
        _tui_health_refresh_task = asyncio.create_task(_tui_health_db_queries(), name="tui-health-db-queries")
        if _tui_health_cache is not None:
            return _build_tui_health_response(_tui_health_cache)
    try:
        db = await asyncio.wait_for(asyncio.shield(_tui_health_refresh_task), timeout=2.0)
        _tui_health_cache = db
        _tui_health_cache_updated_at = utcnow()
    except asyncio.TimeoutError:
        db = _tui_health_cache or {"database": False}
    except Exception:
        _tui_health_refresh_task = None
        db = _tui_health_cache or {"database": False}
    else:
        _tui_health_refresh_task = None
    return _build_tui_health_response(db)


def _get_news_status() -> dict:
    """Get news intelligence status for health check."""
    try:
        from services.news.feed_service import news_feed_service
        from services.news.semantic_matcher import semantic_matcher

        return {
            "enabled": settings.NEWS_EDGE_ENABLED,
            "articles": news_feed_service.article_count,
            "running": news_feed_service._running,
            "matcher": semantic_matcher.get_status(),
        }
    except Exception:
        return {"enabled": False}


def _get_ai_status() -> dict:
    """Get AI status for health check."""
    try:
        from services.ai import get_llm_manager

        manager = get_llm_manager()
        return {"enabled": manager.is_available()}
    except Exception:
        return {"enabled": False}


def _get_ws_feeds_status(
    scanner_status: Optional[dict] = None,
    worker_status: Optional[dict] = None,
) -> dict:
    """Get WebSocket feeds status for health check."""
    if not settings.WS_FEED_ENABLED:
        return {"healthy": False, "started": False, "enabled": False}

    def _first_snapshot_status() -> Optional[dict]:
        if isinstance(scanner_status, dict):
            snapshot_ws = scanner_status.get("ws_feeds")
            if isinstance(snapshot_ws, dict) and snapshot_ws.get("healthy") is True:
                return snapshot_ws

        if isinstance(worker_status, dict):
            crypto_status = worker_status.get("crypto")
            if isinstance(crypto_status, dict):
                crypto_stats = crypto_status.get("stats")
                if isinstance(crypto_stats, dict):
                    snapshot_ws = crypto_stats.get("ws_feeds")
                    if isinstance(snapshot_ws, dict) and snapshot_ws.get("healthy") is True:
                        return snapshot_ws
        return None

    snapshot_ws = _first_snapshot_status()
    if isinstance(snapshot_ws, dict):
        return snapshot_ws

    try:
        from services.ws_feeds import get_feed_manager

        mgr = get_feed_manager()
        return mgr.health_check()
    except Exception:
        return {"healthy": False, "started": False}


def _get_signal_runtime_status() -> dict:
    try:
        from services.runtime_signal_queue import get_queue_depth

        depth = get_queue_depth()
        if isinstance(depth, dict):
            return {
                "depth_by_lane": {str(key): int(value) for key, value in depth.items()},
                "total_depth": sum(int(value) for value in depth.values()),
            }
    except Exception:
        pass
    return {"depth_by_lane": {}, "total_depth": 0}


@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with all system stats"""
    async with AsyncSessionLocal() as session:
        scanner_status = await shared_state.get_scanner_status_from_db(session)
        discovery_status = await discovery_shared_state.get_discovery_status_from_db(session)
        maintenance_row = (
            await session.execute(select(AppSettings).where(AppSettings.id == "default"))
        ).scalar_one_or_none()
        try:
            from services.news import shared_state as news_shared_state

            news_workflow_status = await news_shared_state.get_news_status_from_db(session)
        except Exception:
            news_workflow_status = {}
        worker_status_rows = await list_worker_snapshots(session, include_stats=True, stats_mode="summary")
        orchestrator_snapshot = await read_orchestrator_snapshot(session)
    if isinstance(orchestrator_snapshot, dict):
        orchestrator_snapshot = dict(orchestrator_snapshot)
        orchestrator_snapshot["stats"] = summarize_worker_stats(orchestrator_snapshot.get("stats"))
    worker_status = {row.get("worker_name"): row for row in worker_status_rows}
    maintenance_enabled = bool(settings.AUTO_CLEANUP_ENABLED)
    maintenance_interval = int(settings.CLEANUP_INTERVAL_HOURS)
    if maintenance_row is not None:
        if maintenance_row.auto_cleanup_enabled is not None:
            maintenance_enabled = bool(maintenance_row.auto_cleanup_enabled)
        maintenance_interval = int(maintenance_row.cleanup_interval_hours or maintenance_interval)

    return {
        "status": "healthy",
        "timestamp": utcnow().isoformat(),
        "checks": {
            "database": True,
        },
        "services": {
            "scanner": {
                "running": scanner_status.get("running", False),
                "last_scan": scanner_status.get("last_scan"),
                "opportunities_count": scanner_status.get("opportunities_count", 0),
            },
            "wallet_tracker": {"tracked_wallets": len(await wallet_tracker.get_all_wallets())},
            "trading": {
                "initialized": live_execution_service.is_ready(),
                "stats": live_execution_service.get_stats().__dict__ if live_execution_service.is_ready() else None,
            },
            "trader_orchestrator": {
                "running": bool(orchestrator_snapshot.get("running", False)),
                "stats": orchestrator_snapshot,
            },
            "maintenance": {
                "auto_cleanup_enabled": maintenance_enabled,
                "cleanup_interval_hours": maintenance_interval if maintenance_enabled else None,
            },
            "market_prioritizer": market_prioritizer.get_stats(),
            "ai_intelligence": _get_ai_status(),
            "ws_feeds": _get_ws_feeds_status(scanner_status, worker_status),
            "signal_runtime": {
                "queue_depth": _get_signal_runtime_status(),
            },
            "news_intelligence": _get_news_status(),
            "news_workflow": {
                "running": bool(news_workflow_status.get("running", False)),
                "enabled": bool(news_workflow_status.get("enabled", False)),
                "paused": bool(news_workflow_status.get("paused", False)),
                "last_scan": news_workflow_status.get("last_scan"),
                "next_scan": news_workflow_status.get("next_scan"),
                "current_activity": news_workflow_status.get("current_activity"),
                "last_error": news_workflow_status.get("last_error"),
                "degraded_mode": bool(news_workflow_status.get("degraded_mode", False)),
                "pending_intents": int(news_workflow_status.get("pending_intents", 0)),
            },
            "wallet_discovery": {
                "running": bool(discovery_status.get("running", wallet_discovery._running)),
                "last_run": discovery_status.get("last_run_at")
                or (wallet_discovery._last_run_at.isoformat() if wallet_discovery._last_run_at else None),
                "wallets_discovered": int(
                    discovery_status.get(
                        "wallets_discovered_last_run",
                        wallet_discovery._wallets_discovered_last_run,
                    )
                ),
                "wallets_analyzed": int(
                    discovery_status.get(
                        "wallets_analyzed_last_run",
                        wallet_discovery._wallets_analyzed_last_run,
                    )
                ),
                "current_activity": discovery_status.get("current_activity"),
                "interval_minutes": discovery_status.get("run_interval_minutes"),
                "paused": bool(discovery_status.get("paused", False)),
            },
            "workers": worker_status,
        },
        "rate_limits": rate_limiter.get_status(),
        "api_rate_limit": inbound_api_rate_limiter.status(),
        "config": {
            "scan_interval": settings.SCAN_INTERVAL_SECONDS,
            "min_profit_threshold": settings.MIN_PROFIT_THRESHOLD,
            "max_markets": settings.MAX_MARKETS_TO_SCAN,
            "max_events": settings.MAX_EVENTS_TO_SCAN,
            "market_fetch_page_size": settings.MARKET_FETCH_PAGE_SIZE,
            "market_fetch_order": settings.MARKET_FETCH_ORDER,
        },
    }


# Metrics endpoint (Prometheus format)
@app.get("/metrics")
async def metrics():
    """Prometheus-compatible metrics"""
    async with AsyncSessionLocal() as session:
        scanner_status = await shared_state.get_scanner_status_from_db(session)
        orchestrator_snapshot = await read_orchestrator_snapshot(session)
        try:
            from services.news import shared_state as news_shared_state

            news_status = await news_shared_state.get_news_status_from_db(session)
        except Exception:
            news_status = {}
    opp_count = scanner_status.get("opportunities_count", 0)
    scanner_running = 1 if scanner_status.get("running", False) else 0
    news_stats = news_status.get("stats") or {}
    news_running = 1 if news_status.get("running", False) else 0
    news_degraded = 1 if news_status.get("degraded_mode", False) else 0
    news_last_error = 1 if news_status.get("last_error") else 0
    api_rate_status = inbound_api_rate_limiter.status()
    api_rate_enabled = 1 if api_rate_status.get("enabled") else 0
    api_rate_limit = int(api_rate_status.get("requests_per_window") or 0)
    api_rate_window = int(api_rate_status.get("window_seconds") or 0)
    api_rate_burst = int(api_rate_status.get("burst") or 0)
    api_rate_clients = int(api_rate_status.get("tracked_clients") or 0)

    metrics_text = f"""# HELP polymarket_opportunities_total Total detected opportunities
# TYPE polymarket_opportunities_total gauge
polymarket_opportunities_total {opp_count}

# HELP polymarket_scanner_running Scanner running status
# TYPE polymarket_scanner_running gauge
polymarket_scanner_running {scanner_running}

# HELP polymarket_tracked_wallets Number of tracked wallets
# TYPE polymarket_tracked_wallets gauge
polymarket_tracked_wallets {len(await wallet_tracker.get_all_wallets())}

# HELP polymarket_trader_orchestrator_running Trader orchestrator running status
# TYPE polymarket_trader_orchestrator_running gauge
polymarket_trader_orchestrator_running {1 if orchestrator_snapshot.get("running", False) else 0}

# HELP polymarket_trader_orchestrator_orders Total trader orders executed/submitted
# TYPE polymarket_trader_orchestrator_orders counter
polymarket_trader_orchestrator_orders {orchestrator_snapshot.get("orders_count", 0)}

# HELP polymarket_trader_orchestrator_profit Daily trader orchestrator realized pnl
# TYPE polymarket_trader_orchestrator_profit gauge
polymarket_trader_orchestrator_profit {orchestrator_snapshot.get("daily_pnl", 0.0)}

# HELP polymarket_news_workflow_running News workflow worker running status
# TYPE polymarket_news_workflow_running gauge
polymarket_news_workflow_running {news_running}

# HELP polymarket_news_workflow_pending_intents Pending news intents
# TYPE polymarket_news_workflow_pending_intents gauge
polymarket_news_workflow_pending_intents {news_status.get("pending_intents", 0)}

# HELP polymarket_news_workflow_last_findings Last cycle finding count
# TYPE polymarket_news_workflow_last_findings gauge
polymarket_news_workflow_last_findings {news_stats.get("findings", 0)}

# HELP polymarket_news_workflow_last_intents Last cycle intent count
# TYPE polymarket_news_workflow_last_intents gauge
polymarket_news_workflow_last_intents {news_stats.get("intents", 0)}

# HELP polymarket_news_workflow_llm_calls Last cycle LLM call count
# TYPE polymarket_news_workflow_llm_calls gauge
polymarket_news_workflow_llm_calls {news_stats.get("llm_calls_used", 0)}

# HELP polymarket_news_workflow_budget_skips Last cycle budget skip count
# TYPE polymarket_news_workflow_budget_skips gauge
polymarket_news_workflow_budget_skips {news_stats.get("budget_skip_count", news_stats.get("llm_calls_skipped", 0))}

# HELP polymarket_news_workflow_cycle_duration_seconds Last cycle duration in seconds
# TYPE polymarket_news_workflow_cycle_duration_seconds gauge
polymarket_news_workflow_cycle_duration_seconds {news_stats.get("elapsed_seconds", 0)}

# HELP polymarket_news_workflow_degraded_mode News workflow degraded mode flag
# TYPE polymarket_news_workflow_degraded_mode gauge
polymarket_news_workflow_degraded_mode {news_degraded}

# HELP polymarket_news_workflow_last_error_flag News workflow last error flag
# TYPE polymarket_news_workflow_last_error_flag gauge
polymarket_news_workflow_last_error_flag {news_last_error}

# HELP polymarket_api_rate_limit_enabled Inbound API rate limit enabled flag
# TYPE polymarket_api_rate_limit_enabled gauge
polymarket_api_rate_limit_enabled {api_rate_enabled}

# HELP polymarket_api_rate_limit_requests_per_window Inbound API request limit per window
# TYPE polymarket_api_rate_limit_requests_per_window gauge
polymarket_api_rate_limit_requests_per_window {api_rate_limit}

# HELP polymarket_api_rate_limit_window_seconds Inbound API rate limit window seconds
# TYPE polymarket_api_rate_limit_window_seconds gauge
polymarket_api_rate_limit_window_seconds {api_rate_window}

# HELP polymarket_api_rate_limit_burst Inbound API rate limit burst capacity
# TYPE polymarket_api_rate_limit_burst gauge
polymarket_api_rate_limit_burst {api_rate_burst}

# HELP polymarket_api_rate_limit_tracked_clients Number of active inbound rate-limit buckets
# TYPE polymarket_api_rate_limit_tracked_clients gauge
polymarket_api_rate_limit_tracked_clients {api_rate_clients}
"""

    return JSONResponse(content=metrics_text, media_type="text/plain")


# Serve frontend static files (if built)
frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="static")


def kill_port(port: int):
    """Kill any process currently using the given port."""
    import subprocess
    import signal

    try:
        result = subprocess.run(["lsof", "-ti", f":{port}"], capture_output=True, text=True, timeout=5)
        pids = result.stdout.strip()
        if pids:
            for pid_str in pids.split("\n"):
                pid = int(pid_str.strip())
                # Don't kill ourselves
                if pid == os.getpid():
                    continue
                try:
                    os.kill(pid, signal.SIGKILL)
                    logger.info(f"Killed existing process on port {port}", pid=pid)
                except ProcessLookupError:
                    pass
            import time

            time.sleep(0.5)
    except FileNotFoundError:
        # lsof not available, try fuser as fallback
        try:
            result = subprocess.run(["fuser", f"{port}/tcp"], capture_output=True, text=True, timeout=5)
            pids = result.stdout.strip()
            if pids:
                subprocess.run(["fuser", "-k", f"{port}/tcp"], capture_output=True, timeout=5)
                logger.info(f"Killed existing process on port {port}")
                import time

                time.sleep(0.5)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    except (subprocess.TimeoutExpired, ValueError, OSError):
        pass


if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    kill_port(port)
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        # Keep a single worker because background tasks (scanner, wallet
        # tracker, etc.) hold in-process state. CPU-bound work is offloaded
        # to the thread pool executor configured in lifespan() so the event
        # loop stays free to serve API requests.
        timeout_keep_alive=30,
    )
