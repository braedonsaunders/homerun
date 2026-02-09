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
    KALSHI_WS_URL: str = "wss://api.elections.kalshi.com/trade-api/ws/v2"

    # WebSocket Feed Settings
    WS_FEED_ENABLED: bool = True  # Enable real-time WebSocket price feeds
    WS_RECONNECT_MAX_DELAY: float = 60.0  # Max reconnect backoff seconds
    WS_PRICE_STALE_SECONDS: float = 30.0  # Mark prices stale after this
    WS_HEARTBEAT_INTERVAL: float = 15.0  # Ping interval to keep connection alive

    # Scanner Settings
    SCAN_INTERVAL_SECONDS: int = 60
    MIN_PROFIT_THRESHOLD: float = 0.015  # 1.5% minimum profit after fees
    POLYMARKET_FEE: float = 0.02  # 2% winner fee

    # Market Settings
    MAX_MARKETS_TO_SCAN: int = 500
    MIN_LIQUIDITY: float = 1000.0  # Minimum liquidity in USD

    # Opportunity Quality Filters (hard rejection thresholds)
    MIN_LIQUIDITY_HARD: float = 200.0  # Reject opportunities below this liquidity
    MIN_POSITION_SIZE: float = (
        25.0  # Reject if max position < this (absolute profit too small)
    )
    MIN_ABSOLUTE_PROFIT: float = 5.0  # Reject if net profit on max position < this
    MIN_ANNUALIZED_ROI: float = 10.0  # Reject if annualized ROI < this percent
    MAX_RESOLUTION_MONTHS: int = (
        18  # Reject if resolution > this many months away (capital lockup)
    )

    # NegRisk Exhaustivity Thresholds
    # Genuine NegRisk arbitrage is 1-3% (total YES 0.97-0.99).
    # Anything below 0.93 is almost always non-exhaustive outcomes, not mispricing.
    NEGRISK_MIN_TOTAL_YES: float = 0.90  # Hard reject below this
    NEGRISK_WARN_TOTAL_YES: float = 0.97  # Warn below this
    # Maximum ROI that's plausible for real arbitrage (filter stale/invalid data)
    MAX_PLAUSIBLE_ROI: float = 30.0  # >30% ROI is almost certainly a false positive
    # Max number of legs in a multi-leg trade (slippage compounds per leg)
    MAX_TRADE_LEGS: int = 8

    # Settlement Lag Timing
    SETTLEMENT_LAG_MAX_DAYS_TO_RESOLUTION: int = (
        14  # Only detect settlement lag within this window
    )

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

    # Trading VPN/Proxy Configuration
    # Route ONLY trading requests through a VPN proxy (scanning/data unaffected)
    TRADING_PROXY_ENABLED: bool = False  # Enable proxy for trading requests
    TRADING_PROXY_URL: Optional[str] = None  # e.g., socks5://user:pass@host:port
    TRADING_PROXY_VERIFY_SSL: bool = True  # Verify SSL certs through proxy
    TRADING_PROXY_TIMEOUT: float = 30.0  # Timeout for proxied requests
    TRADING_PROXY_REQUIRE_VPN: bool = True  # Block trades if VPN check fails

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

    # Portfolio Risk Management
    PORTFOLIO_MAX_EXPOSURE_USD: float = 5000.0  # Maximum total portfolio exposure
    PORTFOLIO_MAX_PER_CATEGORY_USD: float = 2000.0  # Max exposure per market category
    PORTFOLIO_MAX_PER_EVENT_USD: float = 1000.0  # Max exposure per event
    PORTFOLIO_BANKROLL_USD: float = 10000.0  # Total bankroll for Kelly sizing
    PORTFOLIO_MAX_KELLY_FRACTION: float = 0.05  # Never bet >5% of bankroll per trade

    # BTC/ETH High-Frequency Strategy
    BTC_ETH_HF_ENABLED: bool = True  # Enable BTC/ETH high-freq scanning
    BTC_ETH_HF_PURE_ARB_MAX_COMBINED: float = 0.98  # Max combined for pure arb
    BTC_ETH_HF_DUMP_THRESHOLD: float = 0.05  # Min drop for dump-hedge trigger
    BTC_ETH_HF_THIN_LIQUIDITY_USD: float = 500.0  # Below this = thin book

    # New Market Detection
    NEW_MARKET_DETECTION_ENABLED: bool = True  # Enable new market monitoring
    NEW_MARKET_WINDOW_SECONDS: int = 300  # Markets seen within this = "new"

    # Combinatorial Validation
    COMBINATORIAL_MIN_CONFIDENCE: float = 0.75  # Min LLM confidence for trades
    COMBINATORIAL_HIGH_CONFIDENCE: float = 0.90  # High confidence threshold
    COMBINATORIAL_MIN_ACCURACY: float = 0.70  # Auto-raise threshold if below

    # Maker Mode / Fee Model
    MAKER_MODE_DEFAULT: bool = True  # Use limit orders (maker) by default
    FEE_MODEL_MAKER_MODE: bool = True  # Pass maker_mode=True to fee model

    # Cross-Platform Arbitrage
    CROSS_PLATFORM_ENABLED: bool = True
    KALSHI_API_URL: str = "https://api.elections.kalshi.com/trade-api/v2"

    # Bayesian Cascade Strategy
    BAYESIAN_CASCADE_ENABLED: bool = True  # Enable Bayesian Cascade strategy
    BAYESIAN_MIN_EDGE_PERCENT: float = 5.0  # Min expected-vs-actual diff to flag (%)
    BAYESIAN_PROPAGATION_DEPTH: int = 3  # Max hops through the dependency graph

    # Liquidity Vacuum
    LIQUIDITY_VACUUM_ENABLED: bool = True
    LIQUIDITY_VACUUM_MIN_IMBALANCE_RATIO: float = 5.0
    LIQUIDITY_VACUUM_MIN_DEPTH_USD: float = 100.0

    # Entropy Arbitrage
    ENTROPY_ARB_ENABLED: bool = True
    ENTROPY_ARB_MIN_DEVIATION: float = 0.15

    # Event-Driven Arbitrage
    EVENT_DRIVEN_ENABLED: bool = True

    # Temporal Decay
    TEMPORAL_DECAY_ENABLED: bool = True

    # Correlation Arbitrage
    CORRELATION_ARB_ENABLED: bool = True
    CORRELATION_ARB_MIN_CORRELATION: float = 0.7
    CORRELATION_ARB_MIN_DIVERGENCE: float = 0.05

    # Market Making
    MARKET_MAKING_ENABLED: bool = True
    MARKET_MAKING_SPREAD_BPS: float = 100.0
    MARKET_MAKING_MAX_INVENTORY_USD: float = 500.0

    # Statistical Arbitrage
    STAT_ARB_ENABLED: bool = True
    STAT_ARB_MIN_EDGE: float = 0.05

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
