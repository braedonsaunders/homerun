from pydantic_settings import BaseSettings
from typing import Optional
from pathlib import Path

# Get the directory where this config file is located (backend/)
_BACKEND_DIR = Path(__file__).parent.resolve()
_DEFAULT_DB_PATH = _BACKEND_DIR / "arbitrage.db"


class Settings(BaseSettings):
    # API Base URLs
    GAMMA_API_URL: str = "https://gamma-api.polymarket.com"
    CLOB_API_URL: str = "https://clob.polymarket.com"
    DATA_API_URL: str = "https://data-api.polymarket.com"

    # WebSocket URLs
    CLOB_WS_URL: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

    # Scanner Settings
    SCAN_INTERVAL_SECONDS: int = 60
    MIN_PROFIT_THRESHOLD: float = 0.025  # 2.5% minimum profit after fees
    POLYMARKET_FEE: float = 0.02  # 2% winner fee

    # Market Settings
    MAX_MARKETS_TO_SCAN: int = 500
    MIN_LIQUIDITY: float = 1000.0  # Minimum liquidity in USD

    # Wallet Tracking
    TRACKED_WALLETS: list[str] = []

    # Notifications
    TELEGRAM_BOT_TOKEN: Optional[str] = None
    TELEGRAM_CHAT_ID: Optional[str] = None

    # Database - use absolute path based on backend directory
    DATABASE_URL: str = f"sqlite+aiosqlite:///{_DEFAULT_DB_PATH}"

    # Production Settings
    LOG_LEVEL: str = "INFO"
    CORS_ORIGINS: list[str] = ["*"]

    # Simulation Defaults
    DEFAULT_SIMULATION_CAPITAL: float = 10000.0
    DEFAULT_MAX_POSITION_PCT: float = 10.0
    DEFAULT_SLIPPAGE_BPS: float = 50.0

    # Copy Trading
    COPY_TRADING_POLL_INTERVAL: int = 30
    DEFAULT_COPY_DELAY_SECONDS: int = 5

    # Anomaly Detection
    MIN_TRADES_FOR_ANALYSIS: int = 10
    SUSPICIOUS_WIN_RATE_THRESHOLD: float = 0.95
    MAX_ANOMALY_SCORE_FOR_COPY: float = 0.5

    # API Settings
    API_TIMEOUT_SECONDS: int = 30
    MAX_RETRY_ATTEMPTS: int = 4
    RETRY_BASE_DELAY: float = 1.0

    # Trading Configuration (Polymarket CLOB)
    # Get these from: https://polymarket.com/settings/api-keys
    POLYMARKET_PRIVATE_KEY: Optional[str] = None  # Wallet private key for signing
    POLYMARKET_API_KEY: Optional[str] = None
    POLYMARKET_API_SECRET: Optional[str] = None
    POLYMARKET_API_PASSPHRASE: Optional[str] = None

    # Trading Safety Limits
    TRADING_ENABLED: bool = False  # Must be explicitly enabled
    MAX_TRADE_SIZE_USD: float = 100.0  # Maximum single trade size
    MAX_DAILY_TRADE_VOLUME: float = 1000.0  # Maximum daily trading volume
    MAX_OPEN_POSITIONS: int = 10  # Maximum concurrent open positions
    MIN_ORDER_SIZE_USD: float = 1.0  # Minimum order size

    # Order Settings
    DEFAULT_ORDER_TYPE: str = "GTC"  # GTC (Good Till Cancel) or FOK (Fill Or Kill)
    MAX_SLIPPAGE_PERCENT: float = 2.0  # Maximum acceptable slippage

    # Polygon Network (for on-chain operations)
    POLYGON_RPC_URL: str = "https://polygon-rpc.com"
    POLYGON_WS_URL: str = "wss://polygon-bor-rpc.publicnode.com"
    CHAIN_ID: int = 137  # Polygon mainnet

    # Depth Analysis
    MIN_DEPTH_USD: float = 200.0  # Minimum order book depth to allow trade

    # Token Circuit Breaker (per-token trip mechanism)
    CB_LARGE_TRADE_SHARES: float = 1500.0  # What counts as a large trade
    CB_CONSECUTIVE_TRIGGER: int = 2  # Large trades in window to trip
    CB_DETECTION_WINDOW_SECONDS: int = 30  # Window for rapid trade detection
    CB_TRIP_DURATION_SECONDS: int = 120  # How long to block a tripped token

    # WebSocket Wallet Monitor
    WS_WALLET_MONITOR_ENABLED: bool = True  # Enable real-time wallet monitoring
    MEMPOOL_MODE_ENABLED: bool = False  # Enable pre-confirmation mempool monitoring

    # Per-Market Position Limits
    MAX_PER_MARKET_USD: float = 500.0  # Maximum USD exposure per market

    # Copy Trading — Whale Filtering
    MIN_WHALE_SHARES: float = 10.0  # Ignore whale trades below this share count

    # Fill Monitor
    FILL_MONITOR_POLL_SECONDS: int = 5  # Fill monitor polling interval

    # CSV Trade Logging
    CSV_TRADE_LOG_ENABLED: bool = True  # Enable append-only CSV trade log

    # Database Maintenance
    AUTO_CLEANUP_ENABLED: bool = False  # Enable automatic cleanup
    CLEANUP_INTERVAL_HOURS: int = 24  # Run cleanup every X hours
    CLEANUP_RESOLVED_TRADE_DAYS: int = 30  # Delete resolved trades older than X days
    CLEANUP_OPEN_TRADE_EXPIRY_DAYS: int = 90  # Expire open trades after X days
    CLEANUP_WALLET_TRADE_DAYS: int = 60  # Delete wallet trades older than X days
    CLEANUP_ANOMALY_DAYS: int = 30  # Delete resolved anomalies older than X days

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
