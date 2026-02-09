import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import FastAPI, WebSocket, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pathlib import Path
import traceback

from config import settings
from api import router, handle_websocket
from api.routes_simulation import simulation_router
from api.routes_copy_trading import copy_trading_router
from api.routes_anomaly import anomaly_router
from api.routes_trading import router as trading_router
from api.routes_auto_trader import router as auto_trader_router
from api.routes_maintenance import router as maintenance_router
from api.routes_settings import router as settings_router
from api.routes_ai import router as ai_router
from api.routes_news import router as news_router
from api.routes_discovery import discovery_router
from api.routes_kalshi import router as kalshi_router
from services import scanner, wallet_tracker, polymarket_client
from services.copy_trader import copy_trader
from services.trading import trading_service
from services.auto_trader import auto_trader
from services.wallet_discovery import wallet_discovery
from services.wallet_intelligence import wallet_intelligence
from services.position_monitor import position_monitor
from services.maintenance import maintenance_service
from services.notifier import notifier
from services.opportunity_recorder import opportunity_recorder
from models.database import init_database
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

    try:
        # Initialize database
        await init_database()
        logger.info("Database initialized")

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

        # Start background tasks
        tasks = []

        scan_task = asyncio.create_task(
            scanner.start_continuous_scan(settings.SCAN_INTERVAL_SECONDS)
        )
        tasks.append(scan_task)

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

        # Start wallet discovery engine (background)
        try:
            await wallet_intelligence.initialize()
            discovery_task = asyncio.create_task(
                wallet_discovery.start_background_discovery(interval_minutes=60)
            )
            tasks.append(discovery_task)
            intelligence_task = asyncio.create_task(
                wallet_intelligence.start_background(interval_minutes=30)
            )
            tasks.append(intelligence_task)
            logger.info("Wallet discovery and intelligence services started")
        except Exception as e:
            logger.warning(f"Wallet discovery startup failed (non-critical): {e}")

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

        # Start news intelligence layer (background news fetching + semantic matcher)
        try:
            from services.news.feed_service import news_feed_service
            from services.news.semantic_matcher import semantic_matcher

            semantic_matcher.initialize()
            if settings.NEWS_EDGE_ENABLED:
                await news_feed_service.start(settings.NEWS_SCAN_INTERVAL_SECONDS)
                logger.info(
                    "News intelligence layer started",
                    ml_mode=semantic_matcher.is_ml_mode,
                    interval=settings.NEWS_SCAN_INTERVAL_SECONDS,
                )
            else:
                logger.info("News intelligence layer initialized (scanning disabled)")
        except Exception as e:
            logger.warning(f"News intelligence init failed (non-critical): {e}")

        # Start Telegram notifier (hooks into scanner + auto_trader callbacks)
        await notifier.start()

        # Start opportunity recorder (hooks into scanner, tracks outcomes)
        await opportunity_recorder.start()

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

        await scanner.stop()
        wallet_tracker.stop()
        copy_trader.stop()
        auto_trader.stop()
        wallet_discovery.stop()
        wallet_intelligence.stop()
        position_monitor.stop()
        maintenance_service.stop()
        try:
            from services.fill_monitor import fill_monitor

            fill_monitor.stop()
        except Exception:
            pass
        notifier.stop()
        opportunity_recorder.stop()
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

        await polymarket_client.close()
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
app.include_router(auto_trader_router, prefix="/api", tags=["Auto Trader"])
app.include_router(maintenance_router, prefix="/api", tags=["Maintenance"])
app.include_router(settings_router, prefix="/api", tags=["Settings"])
app.include_router(ai_router, prefix="/api", tags=["AI Intelligence"])
app.include_router(news_router, prefix="/api", tags=["News Intelligence"])
app.include_router(discovery_router, prefix="/api/discovery", tags=["Trader Discovery"])
app.include_router(kalshi_router, prefix="/api", tags=["Kalshi"])


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
    return {"status": "alive", "timestamp": datetime.utcnow().isoformat()}


@app.get("/health/ready")
async def readiness_check():
    """Readiness probe - is the service ready to accept traffic?"""
    checks = {
        "scanner": scanner.is_running,
        "database": True,  # Would check DB connection
        "polymarket_api": True,  # Would check API availability
    }

    all_ready = all(checks.values())

    return {
        "status": "ready" if all_ready else "not_ready",
        "checks": checks,
        "timestamp": datetime.utcnow().isoformat(),
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


@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with all system stats"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "scanner": {
                "running": scanner.is_running,
                "last_scan": (scanner.last_scan.isoformat() + "Z")
                if scanner.last_scan
                else None,
                "opportunities_count": len(scanner.get_opportunities()),
            },
            "wallet_tracker": {
                "tracked_wallets": len(wallet_tracker.get_all_wallets())
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
            "auto_trader": {
                "running": auto_trader._running,
                "mode": auto_trader.config.mode.value,
                "stats": auto_trader.get_stats(),
            },
            "maintenance": {
                "auto_cleanup_enabled": settings.AUTO_CLEANUP_ENABLED,
                "cleanup_interval_hours": settings.CLEANUP_INTERVAL_HOURS
                if settings.AUTO_CLEANUP_ENABLED
                else None,
            },
            "market_prioritizer": market_prioritizer.get_stats(),
            "ai_intelligence": _get_ai_status(),
            "news_intelligence": _get_news_status(),
            "wallet_discovery": {
                "running": wallet_discovery._running,
                "last_run": wallet_discovery._last_run_at.isoformat()
                if wallet_discovery._last_run_at
                else None,
                "wallets_discovered": wallet_discovery._wallets_discovered_last_run,
            },
            "wallet_intelligence": {
                "running": wallet_intelligence._running,
            },
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
    opportunities = scanner.get_opportunities()

    metrics_text = f"""# HELP polymarket_opportunities_total Total detected opportunities
# TYPE polymarket_opportunities_total gauge
polymarket_opportunities_total {len(opportunities)}

# HELP polymarket_scanner_running Scanner running status
# TYPE polymarket_scanner_running gauge
polymarket_scanner_running {1 if scanner.is_running else 0}

# HELP polymarket_tracked_wallets Number of tracked wallets
# TYPE polymarket_tracked_wallets gauge
polymarket_tracked_wallets {len(wallet_tracker.get_all_wallets())}

# HELP polymarket_copy_configs Active copy trading configurations
# TYPE polymarket_copy_configs gauge
polymarket_copy_configs {len(copy_trader._active_configs)}

# HELP polymarket_trading_enabled Trading enabled status
# TYPE polymarket_trading_enabled gauge
polymarket_trading_enabled {1 if settings.TRADING_ENABLED else 0}

# HELP polymarket_auto_trader_running Auto trader running status
# TYPE polymarket_auto_trader_running gauge
polymarket_auto_trader_running {1 if auto_trader._running else 0}

# HELP polymarket_auto_trader_trades Total auto trades executed
# TYPE polymarket_auto_trader_trades counter
polymarket_auto_trader_trades {auto_trader.stats.total_trades}

# HELP polymarket_auto_trader_profit Total auto trader profit
# TYPE polymarket_auto_trader_profit gauge
polymarket_auto_trader_profit {auto_trader.stats.total_profit}
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
    uvicorn.run(app, host="0.0.0.0", port=port)
