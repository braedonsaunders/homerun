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
    MIN_LIQUIDITY_HARD: float = 1000.0  # Reject opportunities below this liquidity ($)
    MIN_POSITION_SIZE: float = (
        50.0  # Reject if max position < this (absolute profit too small)
    )
    MIN_ABSOLUTE_PROFIT: float = 10.0  # Reject if net profit on max position < this
    MIN_ANNUALIZED_ROI: float = 10.0  # Reject if annualized ROI < this percent
    MAX_RESOLUTION_MONTHS: int = (
        18  # Reject if resolution > this many months away (capital lockup)
    )

    # NegRisk Exhaustivity Thresholds
    # Genuine NegRisk arbitrage is 1-3% (total YES 0.97-0.99).
    # A total YES below 0.95 almost always indicates non-exhaustive outcomes
    # (unlisted candidates, "Other/Field"), not mispricing. The 5%+ gap is
    # rational pricing of the probability that someone not listed wins.
    NEGRISK_MIN_TOTAL_YES: float = 0.95  # Hard reject below this
    NEGRISK_WARN_TOTAL_YES: float = 0.97  # Warn below this
    # Election/primary markets are especially prone to non-exhaustive outcomes
    # because there are always unlisted candidates. Require higher total YES.
    NEGRISK_ELECTION_MIN_TOTAL_YES: float = 0.97  # Hard reject elections below this
    # Maximum spread between earliest and latest resolution dates in a bundle.
    # Mismatched dates can create a gap where ALL outcomes resolve NO.
    NEGRISK_MAX_RESOLUTION_SPREAD_DAYS: int = 7  # Reject if dates differ by more
    # Maximum ROI that's plausible for real arbitrage (filter stale/invalid data)
    MAX_PLAUSIBLE_ROI: float = 30.0  # >30% ROI is almost certainly a false positive
    # Max number of legs in a multi-leg trade (slippage compounds per leg)
    MAX_TRADE_LEGS: int = 6
    # Minimum liquidity per leg: total_liquidity must exceed this * num_legs
    MIN_LIQUIDITY_PER_LEG: float = 500.0  # $500 per leg minimum

    # Settlement Lag Timing
    SETTLEMENT_LAG_MAX_DAYS_TO_RESOLUTION: int = (
        14  # Only detect settlement lag within this window
    )
    SETTLEMENT_LAG_NEAR_ZERO: float = 0.02  # Price below this suggests resolved to NO
    SETTLEMENT_LAG_NEAR_ONE: float = 0.95  # Price above this suggests resolved to YES
    SETTLEMENT_LAG_MIN_SUM_DEVIATION: float = 0.03  # Min deviation from 1.0

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
    # Polymarket series IDs for crypto up-or-down markets (editable in Settings)
    BTC_ETH_HF_SERIES_BTC_15M: str = "10192"
    BTC_ETH_HF_SERIES_ETH_15M: str = "10191"
    BTC_ETH_HF_SERIES_SOL_15M: str = "10423"
    BTC_ETH_HF_SERIES_XRP_15M: str = "10422"
    BTC_ETH_HF_MAKER_MODE: bool = True  # Place maker (limit) orders to avoid fees & earn rebates

    # Miracle Strategy Thresholds
    MIRACLE_MIN_NO_PRICE: float = 0.90  # Only consider NO prices >= this
    MIRACLE_MAX_NO_PRICE: float = 0.999  # Skip if NO already at this+
    MIRACLE_MIN_IMPOSSIBILITY_SCORE: float = 0.70  # Min confidence event is impossible

    # Risk Scoring Thresholds
    RISK_VERY_SHORT_DAYS: int = 2
    RISK_SHORT_DAYS: int = 7
    RISK_LONG_LOCKUP_DAYS: int = 180
    RISK_EXTENDED_LOCKUP_DAYS: int = 90
    RISK_LOW_LIQUIDITY: float = 1000.0
    RISK_MODERATE_LIQUIDITY: float = 5000.0
    RISK_COMPLEX_LEGS: int = 5
    RISK_MULTIPLE_LEGS: int = 3

    # New Market Detection
    NEW_MARKET_DETECTION_ENABLED: bool = True  # Enable new market monitoring
    NEW_MARKET_WINDOW_SECONDS: int = 300  # Markets seen within this = "new"

    # Tiered Scanning (smart market prioritization)
    TIERED_SCANNING_ENABLED: bool = True  # Enable tiered scan loop
    FAST_SCAN_INTERVAL_SECONDS: int = 15  # Hot-tier poll frequency
    FULL_SCAN_INTERVAL_SECONDS: int = 120  # Full (baseline) scan frequency
    HOT_TIER_MAX_AGE_SECONDS: int = 300  # Markets younger than this = HOT
    WARM_TIER_MAX_AGE_SECONDS: int = 1800  # Markets younger than this = WARM
    COLD_TIER_UNCHANGED_CYCLES: int = 5  # Consecutive unchanged cycles before COLD
    THIN_BOOK_LIQUIDITY_THRESHOLD: float = 500.0  # Below this = thin book (HOT signal)
    CRYPTO_PREDICTION_WINDOW_SECONDS: int = (
        30  # Pre-position this far before predicted creation
    )
    INCREMENTAL_FETCH_ENABLED: bool = (
        True  # Use delta fetching for new market detection
    )

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
    ENTROPY_ARB_MIN_DEVIATION: float = 0.25

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

    # News Edge Strategy
    NEWS_EDGE_ENABLED: bool = True  # Enable news-driven edge scanning
    NEWS_SCAN_INTERVAL_SECONDS: int = 60  # Poll news sources every 60s (was 180s)
    NEWS_MIN_EDGE_PERCENT: float = 8.0  # Minimum edge to generate opportunity
    NEWS_MIN_CONFIDENCE: float = 0.6  # Minimum model confidence
    NEWS_MAX_ARTICLES_PER_SCAN: int = 200  # Max articles to process per scan
    NEWS_SIMILARITY_THRESHOLD: float = 0.45  # Cosine similarity threshold for matching
    NEWS_ARTICLE_TTL_HOURS: int = 168  # Keep articles for 7 days (168h)
    NEWS_MAX_OPPORTUNITIES_PER_SCAN: int = 20  # Cap opportunities per scan
    NEWS_GDELT_ENABLED: bool = True  # Enable GDELT as additional news source
    NEWS_RSS_FEEDS: list[str] = []  # Additional custom RSS feed URLs

    # Weather workflow defaults (DB settings override these at runtime)
    WEATHER_WORKFLOW_ENABLED: bool = True
    WEATHER_WORKFLOW_SCAN_INTERVAL_SECONDS: int = 14400
    WEATHER_WORKFLOW_ENTRY_MAX_PRICE: float = 0.25
    WEATHER_WORKFLOW_TAKE_PROFIT_PRICE: float = 0.85
    WEATHER_WORKFLOW_STOP_LOSS_PCT: float = 50.0
    WEATHER_WORKFLOW_MIN_EDGE_PERCENT: float = 8.0
    WEATHER_WORKFLOW_MIN_CONFIDENCE: float = 0.6
    WEATHER_WORKFLOW_MIN_MODEL_AGREEMENT: float = 0.75
    WEATHER_WORKFLOW_MIN_LIQUIDITY: float = 500.0
    WEATHER_WORKFLOW_MAX_MARKETS_PER_SCAN: int = 200
    WEATHER_WORKFLOW_DEFAULT_SIZE_USD: float = 10.0
    WEATHER_WORKFLOW_MAX_SIZE_USD: float = 50.0

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


async def apply_search_filters():
    """Load search filter values from the DB and apply them to the running config singleton.

    Called at startup after DB init, and whenever search filter settings are updated via the API.
    """
    from models.database import AsyncSessionLocal, AppSettings
    from sqlalchemy import select

    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(AppSettings).where(AppSettings.id == "default")
        )
        db = result.scalar_one_or_none()
        if not db:
            return  # No saved settings yet — use defaults

    # Map DB columns → config singleton attributes
    _apply = [
        ("MIN_LIQUIDITY_HARD", "min_liquidity_hard", 1000.0),
        ("MIN_POSITION_SIZE", "min_position_size", 50.0),
        ("MIN_ABSOLUTE_PROFIT", "min_absolute_profit", 10.0),
        ("MIN_ANNUALIZED_ROI", "min_annualized_roi", 10.0),
        ("MAX_RESOLUTION_MONTHS", "max_resolution_months", 18),
        ("MAX_PLAUSIBLE_ROI", "max_plausible_roi", 30.0),
        ("MAX_TRADE_LEGS", "max_trade_legs", 6),
        ("MIN_LIQUIDITY_PER_LEG", "min_liquidity_per_leg", 500.0),
        ("NEGRISK_MIN_TOTAL_YES", "negrisk_min_total_yes", 0.95),
        ("NEGRISK_WARN_TOTAL_YES", "negrisk_warn_total_yes", 0.97),
        ("NEGRISK_ELECTION_MIN_TOTAL_YES", "negrisk_election_min_total_yes", 0.97),
        ("NEGRISK_MAX_RESOLUTION_SPREAD_DAYS", "negrisk_max_resolution_spread_days", 7),
        (
            "SETTLEMENT_LAG_MAX_DAYS_TO_RESOLUTION",
            "settlement_lag_max_days_to_resolution",
            14,
        ),
        ("SETTLEMENT_LAG_NEAR_ZERO", "settlement_lag_near_zero", 0.02),
        ("SETTLEMENT_LAG_NEAR_ONE", "settlement_lag_near_one", 0.95),
        ("SETTLEMENT_LAG_MIN_SUM_DEVIATION", "settlement_lag_min_sum_deviation", 0.03),
        ("BTC_ETH_HF_PURE_ARB_MAX_COMBINED", "btc_eth_pure_arb_max_combined", 0.98),
        ("BTC_ETH_HF_DUMP_THRESHOLD", "btc_eth_dump_hedge_drop_pct", 0.05),
        ("BTC_ETH_HF_THIN_LIQUIDITY_USD", "btc_eth_thin_liquidity_usd", 500.0),
        ("BTC_ETH_HF_SERIES_BTC_15M", "btc_eth_hf_series_btc_15m", "10192"),
        ("BTC_ETH_HF_SERIES_ETH_15M", "btc_eth_hf_series_eth_15m", "10191"),
        ("BTC_ETH_HF_SERIES_SOL_15M", "btc_eth_hf_series_sol_15m", "10423"),
        ("BTC_ETH_HF_SERIES_XRP_15M", "btc_eth_hf_series_xrp_15m", "10422"),
        # Miracle strategy
        ("MIRACLE_MIN_NO_PRICE", "miracle_min_no_price", 0.90),
        ("MIRACLE_MAX_NO_PRICE", "miracle_max_no_price", 0.999),
        ("MIRACLE_MIN_IMPOSSIBILITY_SCORE", "miracle_min_impossibility_score", 0.70),
        # Risk scoring
        ("RISK_VERY_SHORT_DAYS", "risk_very_short_days", 2),
        ("RISK_SHORT_DAYS", "risk_short_days", 7),
        ("RISK_LONG_LOCKUP_DAYS", "risk_long_lockup_days", 180),
        ("RISK_EXTENDED_LOCKUP_DAYS", "risk_extended_lockup_days", 90),
        ("RISK_LOW_LIQUIDITY", "risk_low_liquidity", 1000.0),
        ("RISK_MODERATE_LIQUIDITY", "risk_moderate_liquidity", 5000.0),
        ("RISK_COMPLEX_LEGS", "risk_complex_legs", 5),
        ("RISK_MULTIPLE_LEGS", "risk_multiple_legs", 3),
        # Strategy enable/disable and strategy-specific thresholds
        ("BTC_ETH_HF_ENABLED", "btc_eth_hf_enabled", True),
        ("CROSS_PLATFORM_ENABLED", "cross_platform_enabled", True),
        ("COMBINATORIAL_MIN_CONFIDENCE", "combinatorial_min_confidence", 0.75),
        ("COMBINATORIAL_HIGH_CONFIDENCE", "combinatorial_high_confidence", 0.90),
        ("BAYESIAN_CASCADE_ENABLED", "bayesian_cascade_enabled", True),
        ("BAYESIAN_MIN_EDGE_PERCENT", "bayesian_min_edge_percent", 5.0),
        ("BAYESIAN_PROPAGATION_DEPTH", "bayesian_propagation_depth", 3),
        ("LIQUIDITY_VACUUM_ENABLED", "liquidity_vacuum_enabled", True),
        (
            "LIQUIDITY_VACUUM_MIN_IMBALANCE_RATIO",
            "liquidity_vacuum_min_imbalance_ratio",
            5.0,
        ),
        ("LIQUIDITY_VACUUM_MIN_DEPTH_USD", "liquidity_vacuum_min_depth_usd", 100.0),
        ("ENTROPY_ARB_ENABLED", "entropy_arb_enabled", True),
        ("ENTROPY_ARB_MIN_DEVIATION", "entropy_arb_min_deviation", 0.25),
        ("EVENT_DRIVEN_ENABLED", "event_driven_enabled", True),
        ("TEMPORAL_DECAY_ENABLED", "temporal_decay_enabled", True),
        ("CORRELATION_ARB_ENABLED", "correlation_arb_enabled", True),
        ("CORRELATION_ARB_MIN_CORRELATION", "correlation_arb_min_correlation", 0.7),
        ("CORRELATION_ARB_MIN_DIVERGENCE", "correlation_arb_min_divergence", 0.05),
        ("MARKET_MAKING_ENABLED", "market_making_enabled", True),
        ("MARKET_MAKING_SPREAD_BPS", "market_making_spread_bps", 100.0),
        ("MARKET_MAKING_MAX_INVENTORY_USD", "market_making_max_inventory_usd", 500.0),
        ("STAT_ARB_ENABLED", "stat_arb_enabled", True),
        ("STAT_ARB_MIN_EDGE", "stat_arb_min_edge", 0.05),
        # Scanner basics (already wired but also reloaded here for consistency)
        ("SCAN_INTERVAL_SECONDS", "scan_interval_seconds", 60),
        ("MIN_PROFIT_THRESHOLD", "min_profit_threshold", None),
        ("MAX_MARKETS_TO_SCAN", "max_markets_to_scan", 500),
        ("MIN_LIQUIDITY", "min_liquidity", 1000.0),
    ]

    for config_attr, db_attr, default in _apply:
        db_val = getattr(db, db_attr, None)
        if db_val is not None:
            # min_profit_threshold is stored as percentage in DB, fraction in config
            if db_attr == "min_profit_threshold":
                db_val = db_val / 100.0
            object.__setattr__(settings, config_attr, db_val)
        elif default is not None:
            object.__setattr__(settings, config_attr, default)
