import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from utils.utcnow import utcnow
from fastapi import FastAPI, WebSocket, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pathlib import Path
from typing import Optional
import traceback

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

from config import settings
from api import router, handle_websocket
from api.routes_simulation import simulation_router
from api.routes_copy_trading import copy_trading_router
from api.routes_anomaly import anomaly_router
from api.routes_trading import router as trading_router
from api.routes_maintenance import router as maintenance_router
from api.routes_settings import router as settings_router
from api.routes_ai import router as ai_router
from api.routes_news import router as news_router
from api.routes_discovery import discovery_router
from api.routes_kalshi import router as kalshi_router
from api.routes_plugins import router as plugins_router
from api.routes_crypto import router as crypto_router
from api.routes_news_workflow import router as news_workflow_router
from api.routes_weather_workflow import router as weather_workflow_router
from api.routes_world_intelligence import router as world_intelligence_router
from api.routes_signals import router as signals_router
from api.routes_workers import router as workers_router
from api.routes_validation import router as validation_router
from api.routes_trader_orchestrator import router as trader_orchestrator_router
from api.routes_trader_sources import router as trader_sources_router
from api.routes_traders import router as traders_router
from services import wallet_tracker
from services.copy_trader import copy_trader
from services.trading import trading_service
from services.wallet_discovery import wallet_discovery
from services.position_monitor import position_monitor
from services.maintenance import maintenance_service
from services.validation_service import validation_service
from services.snapshot_broadcaster import snapshot_broadcaster
from models.database import AsyncSessionLocal, init_database
from services import discovery_shared_state, shared_state
from services.news import shared_state as news_shared_state
from services.pause_state import global_pause_state
from services.trader_orchestrator_state import (
    read_orchestrator_control,
    read_orchestrator_snapshot,
)
from services.weather import shared_state as weather_shared_state
from services.worker_state import list_worker_snapshots, read_worker_control
from utils.logger import setup_logging, get_logger
from utils.rate_limiter import rate_limiter

# Import new service modules so their SQLAlchemy models are registered
# before init_database() calls create_all()
from services.depth_analyzer import DepthCheck  # noqa: F401
from services.wallet_ws_monitor import WalletMonitorEvent  # noqa: F401
from services.execution_tiers import TierAssignment  # noqa: F401
from services.price_chaser import OrderRetryLog  # noqa: F401
from services.token_circuit_breaker import TokenTrip  # noqa: F401
from services.category_buffers import CategoryBufferLog  # noqa: F401
from services.market_cache import CachedMarket, CachedUsername  # noqa: F401
from services.market_prioritizer import market_prioritizer  # noqa: F401
from services.live_market_detector import MarketLiveStatus  # noqa: F401
from services.credential_manager import StoredCredential  # noqa: F401
from services.latency_tracker import PipelineLatencyLog  # noqa: F401
from services.sport_classifier import SportTokenClassification  # noqa: F401
from services.fill_monitor import FillEvent  # noqa: F401

# Setup logging
setup_logging(level=settings.LOG_LEVEL if hasattr(settings, "LOG_LEVEL") else "INFO")
logger = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    logger.info("Starting Autonomous Prediction Market Trading Platform...")

    # Create a shared thread pool executor for CPU-bound work so that
    # heavy background tasks (strategy detection, ML embedding, wallet
    # analysis) never block the async event loop that serves API requests.
    import os

    cpu_count = os.cpu_count() or 4
    cpu_executor = ThreadPoolExecutor(
        max_workers=max(cpu_count * 2, 8),
        thread_name_prefix="cpu-pool",
    )
    loop = asyncio.get_running_loop()
    loop.set_default_executor(cpu_executor)
    logger.info(
        "Thread pool executor configured",
        max_workers=max(cpu_count * 2, 8),
    )

    try:
        # Initialize database
        await init_database()
        logger.info("Database initialized")

        # Load DB-backed country reference catalog into runtime cache so
        # country normalization uses dynamic records instead of static files.
        try:
            from services.world_intelligence.country_reference_source import (
                load_country_reference_from_db,
            )

            async with AsyncSessionLocal() as session:
                loaded_country_rows = await load_country_reference_from_db(session)
            logger.info(
                "World country reference loaded",
                rows=loaded_country_rows,
            )
        except Exception as e:
            logger.warning(f"Failed to load world country reference catalog: {e}")

        # Load DB-backed UCDP conflict lists for instability scoring floors.
        try:
            from services.world_intelligence.ucdp_conflict_source import (
                load_ucdp_conflict_lists_from_db,
            )

            async with AsyncSessionLocal() as session:
                ucdp_status = await load_ucdp_conflict_lists_from_db(session)
            logger.info(
                "World UCDP conflict lists loaded",
                source=ucdp_status.get("source"),
                active_wars=ucdp_status.get("active_wars"),
                minor_conflicts=ucdp_status.get("minor_conflicts"),
            )
        except Exception as e:
            logger.warning(f"Failed to load UCDP conflict lists: {e}")

        # Restore global pause state from persisted worker controls.
        # This keeps API-owned loops (copy trader, wallet tracker, LLM/trading gates)
        # aligned with worker controls across restarts.
        try:
            async with AsyncSessionLocal() as session:
                scanner_control = await shared_state.read_scanner_control(session)
                news_control = await news_shared_state.read_news_control(session)
                weather_control = await weather_shared_state.read_weather_control(session)
                discovery_control = await discovery_shared_state.read_discovery_control(
                    session
                )
                orchestrator_control = await read_orchestrator_control(session)
                crypto_control = await read_worker_control(session, "crypto")
                tracked_control = await read_worker_control(session, "tracked_traders")
                world_intel_control = await read_worker_control(
                    session, "world_intelligence"
                )

            should_pause = all(
                bool(control.get("is_paused", False))
                for control in (
                    scanner_control,
                    news_control,
                    weather_control,
                    discovery_control,
                    orchestrator_control,
                    crypto_control,
                    tracked_control,
                    world_intel_control,
                )
            )
            if should_pause:
                global_pause_state.pause()
            else:
                global_pause_state.resume()
            logger.info("Global pause state restored", paused=should_pause)
        except Exception as e:
            logger.warning(f"Failed to restore global pause state (continuing): {e}")

        # Load persisted search filter settings from DB into config singleton
        try:
            from config import apply_search_filters

            await apply_search_filters()
            logger.info("Search filter settings loaded from database")
        except Exception as e:
            logger.warning(
                f"Failed to load search filter settings (using defaults): {e}"
            )

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
                logger.info(
                    "AI intelligence layer initialized (no providers configured)"
                )
        except Exception as e:
            logger.warning(f"AI initialization failed (non-critical): {e}")

        # Add any preconfigured wallets
        for wallet in settings.TRACKED_WALLETS:
            await wallet_tracker.add_wallet(wallet)

        # Background tasks (scanner runs in separate worker process; API reads from DB)
        tasks = []

        # Broadcast scanner snapshot deltas from DB to connected WebSocket clients.
        await snapshot_broadcaster.start(interval_seconds=1.0)

        wallet_task = asyncio.create_task(wallet_tracker.start_monitoring(30))
        tasks.append(wallet_task)

        # Start copy trading service
        await copy_trader.start()

        # Start position monitor (spread trading exit strategies)
        await position_monitor.start()

        # Start fill monitor (read-only, zero risk)
        try:
            from services.fill_monitor import fill_monitor

            await fill_monitor.start()
        except Exception as e:
            logger.warning(f"Fill monitor start failed (non-critical): {e}")

        # Initialize trading service if configured
        if settings.TRADING_ENABLED:
            trading_initialized = await trading_service.initialize()
            if trading_initialized:
                logger.info("Trading service initialized")
            else:
                logger.warning(
                    "Trading service initialization failed - check credentials"
                )

        # Intelligence, crypto, tracked-trader, and trader-orchestrator runtimes are worker-owned.
        logger.info("API runtime running in orchestration/read-only mode for worker-owned loops")

        # Start background cleanup if enabled
        if settings.AUTO_CLEANUP_ENABLED:
            cleanup_config = {
                "resolved_trade_days": settings.CLEANUP_RESOLVED_TRADE_DAYS,
                "open_trade_expiry_days": settings.CLEANUP_OPEN_TRADE_EXPIRY_DAYS,
                "wallet_trade_days": settings.CLEANUP_WALLET_TRADE_DAYS,
                "anomaly_days": settings.CLEANUP_ANOMALY_DAYS,
            }
            cleanup_task = asyncio.create_task(
                maintenance_service.start_background_cleanup(
                    interval_hours=settings.CLEANUP_INTERVAL_HOURS,
                    cleanup_config=cleanup_config,
                )
            )
            tasks.append(cleanup_task)
            logger.info("Background database cleanup enabled")

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
        logger.critical(
            "Startup failed", error=str(e), traceback=traceback.format_exc()
        )
        raise

    finally:
        # Cleanup
        logger.info("Shutting down...")

        await snapshot_broadcaster.stop()
        wallet_tracker.stop()
        copy_trader.stop()
        wallet_discovery.stop()
        position_monitor.stop()
        maintenance_service.stop()
        try:
            from services.fill_monitor import fill_monitor

            fill_monitor.stop()
        except Exception:
            pass
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
        error=str(exc),
        traceback=traceback.format_exc(),
    )
    return JSONResponse(
        status_code=500, content={"detail": "Internal server error", "error": str(exc)}
    )


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS if hasattr(settings, "CORS_ORIGINS") else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Total-Count"],
)


# API routes
app.include_router(router, prefix="/api")
app.include_router(simulation_router, prefix="/api/simulation", tags=["Simulation"])
app.include_router(
    copy_trading_router, prefix="/api/copy-trading", tags=["Copy Trading"]
)
app.include_router(anomaly_router, prefix="/api/anomaly", tags=["Anomaly Detection"])
app.include_router(trading_router, prefix="/api", tags=["Trading"])
app.include_router(trader_orchestrator_router, prefix="/api", tags=["Trader Orchestrator"])
app.include_router(traders_router, prefix="/api", tags=["Traders"])
app.include_router(trader_sources_router, prefix="/api", tags=["Trader Sources"])
app.include_router(maintenance_router, prefix="/api", tags=["Maintenance"])
app.include_router(settings_router, prefix="/api", tags=["Settings"])
app.include_router(ai_router, prefix="/api", tags=["AI Intelligence"])
app.include_router(news_router, prefix="/api", tags=["News Intelligence"])
app.include_router(discovery_router, prefix="/api/discovery", tags=["Trader Discovery"])
app.include_router(kalshi_router, prefix="/api", tags=["Kalshi"])
app.include_router(plugins_router, prefix="/api", tags=["Plugins"])
app.include_router(crypto_router, prefix="/api", tags=["Crypto Markets"])
app.include_router(news_workflow_router, prefix="/api", tags=["News Workflow"])
app.include_router(weather_workflow_router, prefix="/api", tags=["Weather Workflow"])
app.include_router(signals_router, prefix="/api", tags=["Signals"])
app.include_router(workers_router, prefix="/api", tags=["Workers"])
app.include_router(validation_router, prefix="/api", tags=["Validation"])
app.include_router(world_intelligence_router, prefix="/api", tags=["World Intelligence"])


# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
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
        "polymarket_api": True,
    }

    all_ready = all(checks.values())

    return {
        "status": "ready" if all_ready else "not_ready",
        "checks": checks,
        "timestamp": utcnow().isoformat(),
    }


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


def _get_ws_feeds_status(scanner_status: Optional[dict] = None) -> dict:
    """Get WebSocket feeds status for health check."""
    if isinstance(scanner_status, dict):
        snapshot_ws = scanner_status.get("ws_feeds")
        if isinstance(snapshot_ws, dict):
            # Scanner worker owns live WS feed lifecycle in production runs.
            return snapshot_ws
    try:
        from services.ws_feeds import get_feed_manager

        mgr = get_feed_manager()
        return mgr.health_check()
    except Exception:
        return {"healthy": False, "started": False}


@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with all system stats"""
    async with AsyncSessionLocal() as session:
        scanner_status = await shared_state.get_scanner_status_from_db(session)
        discovery_status = await discovery_shared_state.get_discovery_status_from_db(
            session
        )
        try:
            from services.news import shared_state as news_shared_state

            news_workflow_status = await news_shared_state.get_news_status_from_db(
                session
            )
        except Exception:
            news_workflow_status = {}
        worker_status_rows = await list_worker_snapshots(session)
        orchestrator_snapshot = await read_orchestrator_snapshot(session)
    worker_status = {row.get("worker_name"): row for row in worker_status_rows}

    return {
        "status": "healthy",
        "timestamp": utcnow().isoformat(),
        "services": {
            "scanner": {
                "running": scanner_status.get("running", False),
                "last_scan": scanner_status.get("last_scan"),
                "opportunities_count": scanner_status.get("opportunities_count", 0),
            },
            "wallet_tracker": {
                "tracked_wallets": len(await wallet_tracker.get_all_wallets())
            },
            "copy_trader": {
                "running": copy_trader._running,
                "active_configs": len(copy_trader._active_configs),
            },
            "trading": {
                "enabled": settings.TRADING_ENABLED,
                "initialized": trading_service.is_ready(),
                "stats": trading_service.get_stats().__dict__
                if trading_service.is_ready()
                else None,
            },
            "trader_orchestrator": {
                "running": bool(orchestrator_snapshot.get("running", False)),
                "stats": orchestrator_snapshot,
            },
            "maintenance": {
                "auto_cleanup_enabled": settings.AUTO_CLEANUP_ENABLED,
                "cleanup_interval_hours": settings.CLEANUP_INTERVAL_HOURS
                if settings.AUTO_CLEANUP_ENABLED
                else None,
            },
            "market_prioritizer": market_prioritizer.get_stats(),
            "ai_intelligence": _get_ai_status(),
            "ws_feeds": _get_ws_feeds_status(scanner_status),
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
                "running": bool(
                    discovery_status.get("running", wallet_discovery._running)
                ),
                "last_run": discovery_status.get("last_run_at")
                or (
                    wallet_discovery._last_run_at.isoformat()
                    if wallet_discovery._last_run_at
                    else None
                ),
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
        "config": {
            "scan_interval": settings.SCAN_INTERVAL_SECONDS,
            "min_profit_threshold": settings.MIN_PROFIT_THRESHOLD,
            "max_markets": settings.MAX_MARKETS_TO_SCAN,
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

    metrics_text = f"""# HELP polymarket_opportunities_total Total detected opportunities
# TYPE polymarket_opportunities_total gauge
polymarket_opportunities_total {opp_count}

# HELP polymarket_scanner_running Scanner running status
# TYPE polymarket_scanner_running gauge
polymarket_scanner_running {scanner_running}

# HELP polymarket_tracked_wallets Number of tracked wallets
# TYPE polymarket_tracked_wallets gauge
polymarket_tracked_wallets {len(await wallet_tracker.get_all_wallets())}

# HELP polymarket_copy_configs Active copy trading configurations
# TYPE polymarket_copy_configs gauge
polymarket_copy_configs {len(copy_trader._active_configs)}

# HELP polymarket_trading_enabled Trading enabled status
# TYPE polymarket_trading_enabled gauge
polymarket_trading_enabled {1 if settings.TRADING_ENABLED else 0}

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
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"], capture_output=True, text=True, timeout=5
        )
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
            result = subprocess.run(
                ["fuser", f"{port}/tcp"], capture_output=True, text=True, timeout=5
            )
            pids = result.stdout.strip()
            if pids:
                subprocess.run(
                    ["fuser", "-k", f"{port}/tcp"], capture_output=True, timeout=5
                )
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
