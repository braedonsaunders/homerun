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
from services import scanner, wallet_tracker, polymarket_client
from services.simulation import simulation_service
from services.copy_trader import copy_trader
from services.anomaly_detector import anomaly_detector
from services.trading import trading_service
from services.auto_trader import auto_trader
from models.database import init_database
from utils.logger import setup_logging, get_logger
from utils.rate_limiter import rate_limiter

# Setup logging
setup_logging(level=settings.LOG_LEVEL if hasattr(settings, 'LOG_LEVEL') else "INFO")
logger = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    logger.info("Starting Polymarket Arbitrage Scanner...")

    try:
        # Initialize database
        await init_database()
        logger.info("Database initialized")

        # Add any preconfigured wallets
        for wallet in settings.TRACKED_WALLETS:
            await wallet_tracker.add_wallet(wallet)

        # Start background tasks
        tasks = []

        scan_task = asyncio.create_task(
            scanner.start_continuous_scan(settings.SCAN_INTERVAL_SECONDS)
        )
        tasks.append(scan_task)

        wallet_task = asyncio.create_task(
            wallet_tracker.start_monitoring(30)
        )
        tasks.append(wallet_task)

        # Start copy trading service
        await copy_trader.start()

        # Initialize trading service if configured
        if settings.TRADING_ENABLED:
            trading_initialized = await trading_service.initialize()
            if trading_initialized:
                logger.info("Trading service initialized")
            else:
                logger.warning("Trading service initialization failed - check credentials")

        logger.info("All services started successfully")

        yield

    except Exception as e:
        logger.critical("Startup failed", error=str(e), traceback=traceback.format_exc())
        raise

    finally:
        # Cleanup
        logger.info("Shutting down...")

        scanner.stop()
        wallet_tracker.stop()
        copy_trader.stop()
        auto_trader.stop()

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
    lifespan=lifespan
)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(
        "Unhandled exception",
        path=request.url.path,
        method=request.method,
        error=str(exc),
        traceback=traceback.format_exc()
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS if hasattr(settings, 'CORS_ORIGINS') else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Total-Count"],
)


# API routes
app.include_router(router, prefix="/api")
app.include_router(simulation_router, prefix="/api/simulation", tags=["Simulation"])
app.include_router(copy_trading_router, prefix="/api/copy-trading", tags=["Copy Trading"])
app.include_router(anomaly_router, prefix="/api/anomaly", tags=["Anomaly Detection"])
app.include_router(trading_router, prefix="/api", tags=["Trading"])
app.include_router(auto_trader_router, prefix="/api", tags=["Auto Trader"])


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
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with all system stats"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "scanner": {
                "running": scanner.is_running,
                "last_scan": scanner.last_scan.isoformat() if scanner.last_scan else None,
                "opportunities_count": len(scanner.get_opportunities())
            },
            "wallet_tracker": {
                "tracked_wallets": len(wallet_tracker.get_all_wallets())
            },
            "copy_trader": {
                "running": copy_trader._running,
                "active_configs": len(copy_trader._active_configs)
            },
            "trading": {
                "enabled": settings.TRADING_ENABLED,
                "initialized": trading_service.is_ready(),
                "stats": trading_service.get_stats().__dict__ if trading_service.is_ready() else None
            },
            "auto_trader": {
                "running": auto_trader._running,
                "mode": auto_trader.config.mode.value,
                "stats": auto_trader.get_stats()
            }
        },
        "rate_limits": rate_limiter.get_status(),
        "config": {
            "scan_interval": settings.SCAN_INTERVAL_SECONDS,
            "min_profit_threshold": settings.MIN_PROFIT_THRESHOLD,
            "max_markets": settings.MAX_MARKETS_TO_SCAN
        }
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

    return JSONResponse(
        content=metrics_text,
        media_type="text/plain"
    )


# Serve frontend static files (if built)
frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
