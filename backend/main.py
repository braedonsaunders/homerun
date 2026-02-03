import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from config import settings
from api import router, handle_websocket
from services import scanner, wallet_tracker, polymarket_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    print("Starting Polymarket Arbitrage Scanner...")

    # Add any preconfigured wallets
    for wallet in settings.TRACKED_WALLETS:
        await wallet_tracker.add_wallet(wallet)

    # Start background tasks
    scan_task = asyncio.create_task(
        scanner.start_continuous_scan(settings.SCAN_INTERVAL_SECONDS)
    )
    wallet_task = asyncio.create_task(
        wallet_tracker.start_monitoring(30)
    )

    yield

    # Cleanup
    print("Shutting down...")
    scanner.stop()
    wallet_tracker.stop()
    scan_task.cancel()
    wallet_task.cancel()
    await polymarket_client.close()


app = FastAPI(
    title="Polymarket Arbitrage Scanner",
    description="Detect arbitrage opportunities on Polymarket prediction markets",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
app.include_router(router, prefix="/api")


# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await handle_websocket(websocket)


# Health check
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "scanner_running": scanner.is_running,
        "last_scan": scanner.last_scan
    }


# Serve frontend static files (if built)
frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
