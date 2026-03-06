from pathlib import Path
from typing import Optional
import warnings

from pydantic import field_validator
from pydantic_settings import BaseSettings

warnings.filterwarnings(
    "ignore",
    message="urllib3 v2 only supports OpenSSL 1.1.1+.*",
)

_RUNTIME_DATABASE_URL_PATH = Path(__file__).resolve().parent / ".runtime" / "database_url"
_DEFAULT_DATABASE_URL_FALLBACK = "postgresql+asyncpg://homerun:homerun@127.0.0.1:5432/homerun"


def _normalize_raw_database_url(value: object) -> str:
    if value is None:
        return ""
    text = str(value).lstrip("\ufeff").strip().strip('"').strip("'")
    if not text:
        return ""
    return text.rstrip("/")


def _read_runtime_database_url() -> Optional[str]:
    try:
        raw = _RUNTIME_DATABASE_URL_PATH.read_text(encoding="utf-8")
    except OSError:
        return None
    normalized = _normalize_raw_database_url(raw)
    return normalized or None


def _default_database_url() -> str:
    return _read_runtime_database_url() or _DEFAULT_DATABASE_URL_FALLBACK


class Settings(BaseSettings):
    # API Base URLs
    GAMMA_API_URL: str = "https://gamma-api.polymarket.com"
    CLOB_API_URL: str = "https://clob.polymarket.com"
    DATA_API_URL: str = "https://data-api.polymarket.com"

    # WebSocket URLs
    CLOB_WS_URL: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    KALSHI_WS_URL: str = "wss://api.elections.kalshi.com/trade-api/ws/v2"
    BINANCE_WS_ENABLED: bool = True
    BINANCE_WS_URL: str = "wss://stream.binance.com:9443/ws/btcusdt@bookTicker/ethusdt@bookTicker/solusdt@bookTicker/xrpusdt@bookTicker"

    # WebSocket Feed Settings
    WS_FEED_ENABLED: bool = True  # Enable real-time WebSocket price feeds
    WS_RECONNECT_MAX_DELAY: float = 60.0  # Max reconnect backoff seconds
    WS_PRICE_STALE_SECONDS: float = 30.0  # UI/discovery staleness budget for cached prices
    WS_EXECUTION_PRICE_STALE_SECONDS: float = 0.1  # Trader execution staleness budget for cached prices
    EXECUTION_MARKET_DATA_MAX_AGE_MS: int = 100  # Hard freshness gate before trader decisions
    WS_HEARTBEAT_INTERVAL: float = 5.0  # Ping interval to keep connection alive
    WS_REDIS_FLUSH_INTERVAL_SECONDS: float = 0.025  # Redis hot-cache flush cadence for live prices
    WS_STRICT_CONTEXT_WARMUP_SECONDS: float = 0.75  # Strict context WS subscribe warmup before evaluation
    SCANNER_STRICT_WS_MAX_AGE_MS: int = 30000  # WS-only max age for scanner-source strict execution

    # Scanner Settings
    SCAN_WATCHDOG_SECONDS: int = 600  # Max seconds before a scan cycle is killed
    SCANNER_HEARTBEAT_INTERVAL_SECONDS: float = 5.0  # Worker heartbeat persistence cadence
    SCANNER_BATCH_QUEUE_LEASE_SECONDS: int = 45  # Aggregator batch lease duration
    SCANNER_BATCH_QUEUE_RETRY_MAX_ATTEMPTS: int = 5  # Drop batch after repeated aggregator failures
    SCANNER_BATCH_QUEUE_RETAIN_HOURS: int = 24  # Retention horizon for processed scanner batches
    SCANNER_BATCH_QUEUE_CLEANUP_INTERVAL_SECONDS: int = 300  # Queue cleanup cadence
    SCANNER_BATCH_QUEUE_MAX_PENDING: int = 200  # Backpressure cap before scanner drops oldest batch
    OPPORTUNITY_AGGREGATOR_HEARTBEAT_INTERVAL_SECONDS: float = 3.0  # Aggregator heartbeat cadence
    OPPORTUNITY_AGGREGATOR_IDLE_SLEEP_SECONDS: float = 0.25  # Idle sleep between queue polls
    MARKET_UNIVERSE_HEARTBEAT_INTERVAL_SECONDS: float = 5.0  # Catalog worker heartbeat cadence
    MARKET_UNIVERSE_REFRESH_INTERVAL_SECONDS: int = 120  # Full market catalog refresh cadence
    MARKET_UNIVERSE_REFRESH_TIMEOUT_SECONDS: int = 300  # Hard timeout for one market catalog refresh
    MARKET_UNIVERSE_INCREMENTAL_ENABLED: bool = True  # Enable incremental universe diff ingestion
    MARKET_UNIVERSE_INCREMENTAL_TIMEOUT_SECONDS: int = 120  # Timeout for one incremental universe sync
    MARKET_UNIVERSE_INCREMENTAL_SINCE_MINUTES: int = 5  # Delta fetch horizon for updated markets
    MARKET_UNIVERSE_INCREMENTAL_MAX_EVENT_SLUGS: int = 250  # Cap slug fanout for event-diff fetch
    MARKET_UNIVERSE_FULL_RECONCILE_INTERVAL_SECONDS: int = 900  # Forced full reconcile cadence
    MARKET_DATA_WORKER_OWNS_WS: bool = True  # Dedicated market-data worker owns WS lifecycle
    MARKET_DATA_HEARTBEAT_INTERVAL_SECONDS: float = 3.0  # Market data worker heartbeat cadence
    MARKET_DATA_REFRESH_INTERVAL_SECONDS: int = 5  # Market data universe + subscription sync cadence
    MARKET_DATA_STALE_POLL_INTERVAL_SECONDS: int = 3  # Gap-filler stale price poll cadence
    MARKET_DATA_STALE_POLL_BATCH_SIZE: int = 400  # Max stale tokens polled per fallback cycle
    MARKET_DATA_WS_SUBSCRIPTION_CAP: int = 8000  # Cap active WS token subscriptions for safety
    MARKET_DATA_PRICE_STREAM_INTERVAL_SECONDS: float = 1.0  # WS price topic stream cadence
    SCANNER_DEGRADE_HEAVY_ON_BACKLOG: bool = True  # Pause heavy lane under queue pressure
    SCANNER_DEGRADE_HEAVY_BACKLOG_THRESHOLD: int = 120  # Queue threshold for heavy-lane degradation
    OPPORTUNITY_AGGREGATOR_BATCH_TIMEOUT_SECONDS: int = 45  # Timeout for one scanner batch aggregation
    OPPORTUNITY_AGGREGATOR_STRATEGY_TIMEOUT_SECONDS: int = 20  # Timeout for one strategy bridge group
    OPPORTUNITY_AGGREGATOR_GLOBAL_RUN_TIMEOUT_SECONDS: int = 120  # Hard deadline for one aggregator loop run
    OPPORTUNITY_AGGREGATOR_STRATEGY_DLQ_RETRY_MAX_ATTEMPTS: int = 10  # Strategy DLQ retry cap before terminal
    OPPORTUNITY_AGGREGATOR_STRATEGY_DLQ_RETAIN_HOURS: int = 72  # Retention for processed strategy DLQ rows
    SCANNER_SLO_WORKER_INTERVAL_SECONDS: int = 5  # Scanner SLO evaluator cadence
    SCANNER_SLO_MAX_FAST_SCAN_AGE_SECONDS: float = 120.0  # SLO threshold for fast scan freshness
    SCANNER_SLO_MAX_FULL_COVERAGE_COMPLETION_SECONDS: float = 900.0  # SLO threshold for full coverage cycle
    SCANNER_SLO_MAX_OPPORTUNITY_PRICE_AGE_P95_SECONDS: float = 60.0  # SLO threshold for price freshness p95
    SCANNER_SLO_MAX_OPPORTUNITY_LAST_DETECTED_AGE_P95_SECONDS: float = 180.0  # SLO threshold detect freshness p95
    SCANNER_SLO_MIN_COVERAGE_RATIO: float = 0.95  # SLO minimum coverage ratio
    SCANNER_SLO_DEGRADE_HEAVY_ENABLED: bool = True  # Auto-degrade heavy lane on SLO breach
    SCANNER_SLO_DEGRADE_HEAVY_DURATION_SECONDS: int = 180  # Heavy-lane degrade hold duration
    SCAN_INTERVAL_SECONDS: int = 60
    SCANNER_STALE_OPPORTUNITY_MINUTES: int = 45
    # Prevent one strategy from flooding the opportunities surface.
    # Set to <=0 to disable a given cap.
    SCANNER_MAX_OPPORTUNITIES_TOTAL: int = 500
    SCANNER_MAX_OPPORTUNITIES_PER_STRATEGY: int = 120
    MIN_PROFIT_THRESHOLD: float = 0.025  # 2.5% minimum profit after fees
    POLYMARKET_FEE: float = 0.02  # 2% winner fee

    # Trader Discovery Worker
    DISCOVERY_RUN_INTERVAL_MINUTES: int = 15
    DISCOVERY_MAX_MARKETS_PER_RUN: int = 100
    DISCOVERY_MAX_WALLETS_PER_MARKET: int = 50

    # Market Settings
    MAX_MARKETS_TO_SCAN: int = 0  # 0 disables cap (scan full active market universe)
    MAX_EVENTS_TO_SCAN: int = 0  # 0 disables cap (scan full active event universe)
    MARKET_FETCH_PAGE_SIZE: int = 200
    MARKET_FETCH_ORDER: str = "volume"  # volume, updatedAt, createdAt, or empty for default
    MIN_LIQUIDITY: float = 1000.0  # Minimum liquidity in USD
    SCANNER_FORCE_FULL_UNIVERSE: bool = True  # Ignore market/event caps for full-coverage discovery

    # Opportunity Quality Filters (hard rejection thresholds)
    MIN_LIQUIDITY_HARD: float = 1000.0  # Reject opportunities below this liquidity ($)
    MIN_POSITION_SIZE: float = 50.0  # Reject if max position < this (absolute profit too small)
    MIN_ABSOLUTE_PROFIT: float = 10.0  # Reject if net profit on max position < this
    MIN_ANNUALIZED_ROI: float = 10.0  # Reject if annualized ROI < this percent
    MAX_RESOLUTION_MONTHS: int = 18  # Reject if resolution > this many months away (capital lockup)
    # Maximum ROI that's plausible for real arbitrage (filter stale/invalid data)
    MAX_PLAUSIBLE_ROI: float = 30.0  # >30% ROI is almost certainly a false positive
    # Max number of legs in a multi-leg trade (slippage compounds per leg)
    MAX_TRADE_LEGS: int = 6
    # Minimum liquidity per leg: total_liquidity must exceed this * num_legs
    MIN_LIQUIDITY_PER_LEG: float = 500.0  # $500 per leg minimum

    # Wallet Tracking
    TRACKED_WALLETS: list[str] = []

    # Notifications
    TELEGRAM_BOT_TOKEN: Optional[str] = None
    TELEGRAM_CHAT_ID: Optional[str] = None

    # Database
    DATABASE_URL: str = _default_database_url()
    DATABASE_POOL_SIZE: int = 12
    DATABASE_MAX_OVERFLOW: int = 8
    DATABASE_WORKER_POOL_SIZE: int = 2
    DATABASE_WORKER_MAX_OVERFLOW: int = 2
    DATABASE_POOL_TIMEOUT_SECONDS: int = 30
    DATABASE_POOL_RECYCLE_SECONDS: int = 300
    DATABASE_CONNECT_TIMEOUT_SECONDS: float = 8.0
    DATABASE_STATEMENT_TIMEOUT_MS: int = 45000
    DATABASE_IDLE_IN_TRANSACTION_TIMEOUT_MS: int = 60000

    # Redis (worker/event IPC)
    REDIS_HOST: str = "127.0.0.1"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    REDIS_CONNECT_TIMEOUT_SECONDS: float = 1.0
    REDIS_SOCKET_TIMEOUT_SECONDS: float = 1.0
    REDIS_STREAM_BLOCK_MS: int = 20
    REDIS_STREAM_READ_COUNT: int = 200
    EVENT_BUS_STREAM_KEY: str = "homerun:event_bus"
    DATA_EVENT_STREAM_KEY: str = "homerun:data_events"
    REDIS_EVENT_STREAM_MAXLEN: int = 50000
    TRADE_SIGNAL_STREAM_KEY: str = "homerun:trade_signals"
    TRADE_SIGNAL_STREAM_GROUP: str = "trader_orchestrator"
    TRADE_SIGNAL_STREAM_MAXLEN: int = 200000
    TRADE_SIGNAL_STREAM_BLOCK_MS: int = 20
    TRADE_SIGNAL_STREAM_READ_COUNT: int = 500
    TRADE_SIGNAL_STREAM_CLAIM_IDLE_MS: int = 2000
    TRADE_SIGNAL_STREAM_CLAIM_READ_COUNT: int = 500
    TRADE_SIGNAL_STREAM_CLAIM_INTERVAL_SECONDS: float = 1.0
    CRYPTO_WS_UPDATE_STREAM_KEY: str = "homerun:crypto_ws_updates"
    CRYPTO_WS_UPDATE_STREAM_GROUP: str = "crypto_worker"
    CRYPTO_WS_UPDATE_STREAM_MAXLEN: int = 200000
    CRYPTO_WS_UPDATE_STREAM_BLOCK_MS: int = 10
    CRYPTO_WS_UPDATE_STREAM_READ_COUNT: int = 1000
    CRYPTO_WS_UPDATE_STREAM_CLAIM_IDLE_MS: int = 1000
    CRYPTO_WS_UPDATE_STREAM_CLAIM_READ_COUNT: int = 1000
    CRYPTO_WS_UPDATE_STREAM_CLAIM_INTERVAL_SECONDS: float = 0.5
    ORCHESTRATOR_MAINTENANCE_INTERVAL_SECONDS: float = 5.0
    EVENT_HANDLER_TIMEOUT_SECONDS: float = 60.0

    CRYPTO_WS_REACTIVE_DEBOUNCE_SECONDS: float = 0.05

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
    API_RATE_LIMIT_ENABLED: bool = True
    API_RATE_LIMIT_REQUESTS_PER_WINDOW: int = 240
    API_RATE_LIMIT_WINDOW_SECONDS: int = 60
    API_RATE_LIMIT_BURST: int = 300

    # Trading Configuration (Polymarket CLOB)
    # Get these from: https://polymarket.com/settings/api-keys
    POLYMARKET_PRIVATE_KEY: Optional[str] = None  # Wallet private key for signing
    POLYMARKET_API_KEY: Optional[str] = None
    POLYMARKET_API_SECRET: Optional[str] = None
    POLYMARKET_API_PASSPHRASE: Optional[str] = None

    # Trading Safety Limits
    MAX_TRADE_SIZE_USD: float = 100.0  # Maximum single trade size
    MAX_DAILY_TRADE_VOLUME: float = 1000.0  # Maximum daily trading volume
    MIN_ORDER_SIZE_USD: float = 1.0  # Minimum order size

    # Order Settings
    DEFAULT_ORDER_TYPE: str = "GTC"  # GTC (Good Till Cancel) or FOK (Fill Or Kill)
    MAX_SLIPPAGE_PERCENT: float = 2.0  # Maximum acceptable slippage

    # Polygon Network (for on-chain operations)
    POLYGON_RPC_URL: str = "https://rpc-mainnet.matic.quiknode.pro"
    POLYGON_WS_URL: str = "wss://polygon-bor-rpc.publicnode.com"
    CHAIN_ID: int = 137  # Polygon mainnet
    POLYMARKET_SIGNATURE_TYPE: int = 1  # 0=EOA, 1=POLY_PROXY, 2=POLY_GNOSIS_SAFE
    POLYMARKET_FUNDER: Optional[str] = None  # Proxy wallet that holds funds for signature types 1/2

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
    # Polymarket series IDs for crypto up-or-down markets (editable in Settings).
    # 15m, 5m, hourly, and 4-hour market series are independently configurable.
    BTC_ETH_HF_SERIES_BTC_15M: str = "10192"
    BTC_ETH_HF_SERIES_ETH_15M: str = "10191"
    BTC_ETH_HF_SERIES_SOL_15M: str = "10423"
    BTC_ETH_HF_SERIES_XRP_15M: str = "10422"
    BTC_ETH_HF_SERIES_BTC_5M: str = "10684"
    BTC_ETH_HF_SERIES_ETH_5M: str = "10683"
    BTC_ETH_HF_SERIES_SOL_5M: str = "10686"
    BTC_ETH_HF_SERIES_XRP_5M: str = "10685"
    BTC_ETH_HF_SERIES_BTC_1H: str = "10114"
    BTC_ETH_HF_SERIES_ETH_1H: str = "10117"
    BTC_ETH_HF_SERIES_SOL_1H: str = "10122"
    BTC_ETH_HF_SERIES_XRP_1H: str = "10123"
    BTC_ETH_HF_SERIES_BTC_4H: str = "10331"
    BTC_ETH_HF_SERIES_ETH_4H: str = "10332"
    BTC_ETH_HF_SERIES_SOL_4H: str = "10333"
    BTC_ETH_HF_SERIES_XRP_4H: str = "10327"
    BTC_ETH_HF_MAKER_MODE: bool = True  # Place maker (limit) orders to avoid fees & earn rebates
    BTC_ETH_HF_FEE_ESTIMATE: float = 0.0156  # Midpoint taker fee estimate at 50% probability

    # Holding Reward Yield Strategy
    HOLDING_REWARD_YIELD_ENABLED: bool = True
    HOLDING_REWARD_MIN_APY: float = 2.0  # Minimum annualized APY (%) to surface
    HOLDING_REWARD_MIN_LIQUIDITY: float = 5000.0  # Min USD liquidity to consider
    HOLDING_REWARD_MIN_DAYS_TO_RESOLUTION: float = 30.0  # Must be long-dated

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
    # Full-snapshot strategies are CPU-heavy on large catalogs. Run them on a
    # slower cadence with an explicit bounded market batch.
    SCANNER_FULL_SNAPSHOT_STRATEGY_INTERVAL_SECONDS: int = 120
    SCANNER_FULL_SNAPSHOT_MAX_MARKETS: int = 0  # 0 disables cap (scan full active market universe)
    SCANNER_FULL_SNAPSHOT_CHUNK_SIZE: int = 300  # Markets processed per heavy-lane chunk
    SCANNER_FAST_STRATEGY_TIMEOUT_SECONDS: float = 12.0
    SCANNER_FULL_SNAPSHOT_STRATEGY_TIMEOUT_SECONDS: float = 60.0
    SCANNER_FULL_SNAPSHOT_WATCHDOG_SECONDS: int = 180
    # Maximum allowed age for live token prices attached to opportunities.
    # Stale opportunities are pruned before UI/autotrader visibility.
    SCANNER_MARKET_PRICE_MAX_AGE_SECONDS: int = 90
    REALTIME_SCAN_DEBOUNCE_SECONDS: float = 0.25  # WS change coalescing window
    REALTIME_SCAN_MAX_PENDING_TOKENS: int = 2000  # Backpressure cap for queued changed tokens
    REALTIME_SCAN_MAX_BATCH_TOKENS: int = 500  # Max changed tokens consumed per fast scan
    REALTIME_SCAN_MAX_BATCH_MARKETS: int = 800  # Max affected markets evaluated per fast scan
    REALTIME_SCAN_HTTP_PRICE_FALLBACK_CAP: int = 200  # Max HTTP token lookups in reactive scans
    # Opportunity card sparkline history (longer-term trend, not tick-level noise)
    SCANNER_SPARKLINE_WINDOW_HOURS: int = 24
    SCANNER_SPARKLINE_SAMPLE_SECONDS: int = 120
    SCANNER_SPARKLINE_MAX_POINTS: int = 960
    SCANNER_SPARKLINE_EXPORT_POINTS: int = 960
    HOT_TIER_MAX_AGE_SECONDS: int = 300  # Markets younger than this = HOT
    WARM_TIER_MAX_AGE_SECONDS: int = 1800  # Markets younger than this = WARM
    COLD_TIER_UNCHANGED_CYCLES: int = 5  # Consecutive unchanged cycles before COLD
    THIN_BOOK_LIQUIDITY_THRESHOLD: float = 500.0  # Below this = thin book (HOT signal)
    CRYPTO_PREDICTION_WINDOW_SECONDS: int = 30  # Pre-position this far before predicted creation
    INCREMENTAL_FETCH_ENABLED: bool = True  # Use delta fetching for new market detection
    WS_PRICE_HISTORY_MAX_SNAPSHOTS: int = 1500  # Per-token in-memory ring buffer for strategy SDK history

    # Maker Mode / Fee Model
    MAKER_MODE_DEFAULT: bool = True  # Use limit orders (maker) by default
    FEE_MODEL_MAKER_MODE: bool = True  # Pass maker_mode=True to fee model

    # Cross-platform market API
    KALSHI_API_URL: str = "https://api.elections.kalshi.com/trade-api/v2"

    # Entropy Arbitrage
    ENTROPY_ARB_ENABLED: bool = True
    ENTROPY_ARB_MIN_DEVIATION: float = 0.25

    # Event-Driven Arbitrage
    EVENT_DRIVEN_ENABLED: bool = True

    # Temporal decay certainty-shock branch (rapid repricing toward near-certain outcome)
    TEMPORAL_SHOCK_ENABLED: bool = True
    TEMPORAL_SHOCK_LOOKBACK_SECONDS: int = 21600  # 6h rolling window
    TEMPORAL_SHOCK_MIN_POINTS: int = 3
    TEMPORAL_SHOCK_MAX_DAYS_TO_DEADLINE: float = 10.0
    TEMPORAL_SHOCK_MIN_DAYS_TO_DEADLINE: float = -0.25
    TEMPORAL_SHOCK_MIN_ABS_MOVE: float = 0.18
    TEMPORAL_SHOCK_MAX_RETRACE: float = 0.12
    TEMPORAL_SHOCK_MIN_FAVORED_PRICE: float = 0.55
    TEMPORAL_SHOCK_MAX_FAVORED_PRICE: float = 0.97
    TEMPORAL_SHOCK_TARGET_CERTAINTY: float = 0.96
    TEMPORAL_SHOCK_EXTENSION_FACTOR: float = 0.45
    TEMPORAL_SHOCK_MIN_EXPECTED_MOVE: float = 0.03
    TEMPORAL_SHOCK_MIN_LIQUIDITY_HARD: float = 1000.0
    TEMPORAL_SHOCK_MIN_POSITION_SIZE: float = 50.0

    # News Edge Strategy
    NEWS_EDGE_ENABLED: bool = True  # Enable news-driven edge scanning
    NEWS_SCAN_INTERVAL_SECONDS: int = 60  # Poll news sources every 60s (was 180s)
    # 5% edge covers fees + slippage and is meaningful in prediction markets.
    # Previous 8% default was too strict and filtered actionable opportunities.
    NEWS_MIN_EDGE_PERCENT: float = 5.0
    # 0.45 allows moderate-confidence news signals through. News edges carry
    # inherent uncertainty; 0.6 was too restrictive for most workflows.
    NEWS_MIN_CONFIDENCE: float = 0.45
    NEWS_MAX_ARTICLES_PER_SCAN: int = 200  # Max articles to process per scan
    # 0.25 casts a wider net for retrieval; the LLM reranker handles
    # false positives.  Previous 0.45 was too aggressive before reranking.
    NEWS_SIMILARITY_THRESHOLD: float = 0.25
    NEWS_ARTICLE_TTL_HOURS: int = 168  # Keep articles for 7 days (168h)
    NEWS_MAX_OPPORTUNITIES_PER_SCAN: int = 20  # Cap opportunities per scan
    NEWS_GDELT_ENABLED: bool = True  # Enable GDELT as additional news source
    NEWS_RSS_FEEDS: list[str] = []  # Additional custom RSS feed URLs
    NEWS_GOV_RSS_ENABLED: bool = True  # Enable government RSS feed ingestion

    # Weather workflow defaults (DB settings override these at runtime)
    WEATHER_WORKFLOW_ENABLED: bool = True
    WEATHER_WORKFLOW_SCAN_INTERVAL_SECONDS: int = 14400
    WEATHER_WORKFLOW_ENTRY_MAX_PRICE: float = 0.92
    WEATHER_WORKFLOW_TAKE_PROFIT_PRICE: float = 0.85
    WEATHER_WORKFLOW_STOP_LOSS_PCT: float = 50.0
    WEATHER_WORKFLOW_MIN_EDGE_PERCENT: float = 2.0
    WEATHER_WORKFLOW_MIN_CONFIDENCE: float = 0.3
    WEATHER_WORKFLOW_MIN_MODEL_AGREEMENT: float = 0.75
    WEATHER_WORKFLOW_MIN_LIQUIDITY: float = 500.0
    WEATHER_WORKFLOW_MAX_MARKETS_PER_SCAN: int = 200
    WEATHER_WORKFLOW_DEFAULT_SIZE_USD: float = 10.0
    WEATHER_WORKFLOW_MAX_SIZE_USD: float = 50.0

    # Events Settings
    EVENTS_ENABLED: bool = True
    EVENTS_INTERVAL_SECONDS: int = 300  # 5 minutes
    EVENTS_EMIT_TRADE_SIGNALS: bool = False  # keep decoupled from trader orchestrator by default
    ACLED_API_KEY: Optional[str] = None  # ACLED API key (free registration)
    ACLED_EMAIL: Optional[str] = None  # Email for ACLED API auth
    OPENSKY_USERNAME: Optional[str] = None  # OpenSky Network credentials (optional)
    OPENSKY_PASSWORD: Optional[str] = None
    AISSTREAM_API_KEY: Optional[str] = None  # AIS stream API key (optional)
    CLOUDFLARE_RADAR_TOKEN: Optional[str] = None  # Cloudflare Radar API token
    EVENTS_AIS_ENABLED: bool = True
    EVENTS_AIS_WS_URL: str = "wss://stream.aisstream.io/v0/stream"
    EVENTS_AIS_SAMPLE_SECONDS: int = 10
    EVENTS_AIS_MAX_MESSAGES: int = 250
    EVENTS_AIRPLANES_LIVE_ENABLED: bool = True
    EVENTS_AIRPLANES_LIVE_URL: str = "https://api.airplanes.live/v2/mil"
    EVENTS_AIRPLANES_LIVE_TIMEOUT_SECONDS: float = 20.0
    EVENTS_AIRPLANES_LIVE_MAX_RECORDS: int = 1500
    EVENTS_COUNTRY_REFERENCE_SYNC_ENABLED: bool = True
    EVENTS_COUNTRY_REFERENCE_SYNC_HOURS: int = 24
    EVENTS_COUNTRY_REFERENCE_REQUEST_TIMEOUT_SECONDS: float = 20.0
    EVENTS_UCDP_SYNC_ENABLED: bool = True
    EVENTS_UCDP_SYNC_HOURS: int = 24
    EVENTS_UCDP_LOOKBACK_YEARS: int = 8
    EVENTS_UCDP_MAX_PAGES: int = 100
    EVENTS_UCDP_REQUEST_TIMEOUT_SECONDS: float = 25.0
    EVENTS_MID_SYNC_ENABLED: bool = True
    EVENTS_MID_SYNC_HOURS: int = 168
    EVENTS_MID_REQUEST_TIMEOUT_SECONDS: float = 20.0
    EVENTS_TRADE_DEPENDENCY_SYNC_ENABLED: bool = True
    EVENTS_TRADE_DEPENDENCY_SYNC_HOURS: int = 24
    EVENTS_TRADE_DEPENDENCY_REQUEST_TIMEOUT_SECONDS: float = 20.0
    EVENTS_TRADE_DEPENDENCY_WB_PER_PAGE: int = 5000
    EVENTS_TRADE_DEPENDENCY_WB_MAX_PAGES: int = 50
    EVENTS_TRADE_DEPENDENCY_BASE_DIVISOR: float = 120.0
    EVENTS_TRADE_DEPENDENCY_MIN_FACTOR: float = 0.5
    EVENTS_TRADE_DEPENDENCY_MAX_FACTOR: float = 1.5
    EVENTS_CONVERGENCE_MIN_TYPES: int = 2  # Min signal types for convergence
    EVENTS_ANOMALY_THRESHOLD: float = 1.8  # Z-score threshold for anomalies
    EVENTS_ANOMALY_MIN_BASELINE_POINTS: int = 3
    EVENTS_INSTABILITY_SIGNAL_MIN: float = 15.0
    EVENTS_INSTABILITY_CRITICAL: float = 60.0  # CII threshold for critical alerts
    EVENTS_TENSION_CRITICAL: float = 70.0  # Tension score threshold for alerts
    EVENTS_TENSION_PAIRS: list[str] = []
    EVENTS_GDELT_QUERY_DELAY_SECONDS: float = 5.0
    EVENTS_GDELT_MAX_CONCURRENCY: int = 1
    EVENTS_GDELT_NEWS_ENABLED: bool = True
    EVENTS_GDELT_NEWS_TIMESPAN_HOURS: int = 6
    EVENTS_GDELT_NEWS_MAX_RECORDS: int = 40
    EVENTS_GDELT_NEWS_REQUEST_TIMEOUT_SECONDS: float = 20.0
    EVENTS_GDELT_NEWS_CACHE_SECONDS: int = 300
    EVENTS_GDELT_NEWS_QUERY_DELAY_SECONDS: float = 5.0
    EVENTS_GDELT_NEWS_SYNC_ENABLED: bool = True
    EVENTS_GDELT_NEWS_SYNC_HOURS: int = 1
    EVENTS_ACLED_RATE_LIMIT_PER_MIN: int = 5
    EVENTS_ACLED_AUTH_RATE_LIMIT_PER_MIN: int = 12
    EVENTS_ACLED_CB_MAX_FAILURES: int = 8
    EVENTS_ACLED_CB_COOLDOWN_SECONDS: float = 180.0
    EVENTS_USGS_MIN_MAGNITUDE: float = 4.5
    EVENTS_GOV_RSS_ENABLED: bool = True  # Legacy flag (deprecated; NEWS_GOV_RSS_ENABLED is authoritative)
    EVENTS_USGS_ENABLED: bool = True  # Enable earthquake monitoring

    # Database Maintenance
    AUTO_CLEANUP_ENABLED: bool = False  # Enable automatic cleanup
    CLEANUP_INTERVAL_HOURS: int = 24  # Run cleanup every X hours
    CLEANUP_RESOLVED_TRADE_DAYS: int = 30  # Delete resolved trades older than X days
    CLEANUP_OPEN_TRADE_EXPIRY_DAYS: int = 90  # Expire open trades after X days
    CLEANUP_WALLET_TRADE_DAYS: int = 60  # Delete wallet trades older than X days
    CLEANUP_ANOMALY_DAYS: int = 30  # Delete resolved anomalies older than X days
    CLEANUP_TRADE_SIGNAL_EMISSION_DAYS: int = 21  # Delete trade signal emission rows older than X days
    CLEANUP_TRADE_SIGNAL_UPDATE_DAYS: int = 3  # Delete noisy upsert_update emissions older than X days
    CLEANUP_WALLET_ACTIVITY_ROLLUP_DAYS: int = 60  # Delete wallet activity rollups older than X days
    CLEANUP_WALLET_ACTIVITY_DEDUPE_ENABLED: bool = True  # Remove duplicate rollups during maintenance

    @field_validator(
        "GAMMA_API_URL",
        "CLOB_API_URL",
        "DATA_API_URL",
        "CLOB_WS_URL",
        "KALSHI_WS_URL",
        "POLYGON_RPC_URL",
        "POLYGON_WS_URL",
        mode="before",
    )
    @classmethod
    def _normalize_url_field(cls, value: object) -> object:
        """Trim accidental quotes/whitespace from URL env vars."""
        if value is None:
            return value
        text = str(value).strip().strip('"').strip("'")
        if not text:
            return text
        # Keep scheme://host normalization simple and deterministic.
        return text.rstrip("/")

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def _normalize_database_url(cls, value: object) -> object:
        """Normalize DB URL from env/CLI input."""
        normalized = _normalize_raw_database_url(value)
        if not normalized and value is None:
            return value
        return normalized

    class Config:
        pass


settings = Settings()

RUNTIME_SETTINGS_PRECEDENCE = "db_non_null_override > env > code_default"


_EVENTS_DB_FIELD_MAP: dict[str, tuple[str, object]] = {
    "enabled": ("EVENTS_ENABLED", True),
    "interval_seconds": ("EVENTS_INTERVAL_SECONDS", 300),
    "emit_trade_signals": ("EVENTS_EMIT_TRADE_SIGNALS", False),
    "ais_enabled": ("EVENTS_AIS_ENABLED", True),
    "ais_sample_seconds": ("EVENTS_AIS_SAMPLE_SECONDS", 10),
    "ais_max_messages": ("EVENTS_AIS_MAX_MESSAGES", 250),
    "airplanes_live_enabled": ("EVENTS_AIRPLANES_LIVE_ENABLED", True),
    "airplanes_live_timeout_seconds": (
        "EVENTS_AIRPLANES_LIVE_TIMEOUT_SECONDS",
        20.0,
    ),
    "airplanes_live_max_records": ("EVENTS_AIRPLANES_LIVE_MAX_RECORDS", 1500),
    "country_reference_sync_enabled": ("EVENTS_COUNTRY_REFERENCE_SYNC_ENABLED", True),
    "country_reference_sync_hours": ("EVENTS_COUNTRY_REFERENCE_SYNC_HOURS", 24),
    "country_reference_request_timeout_seconds": (
        "EVENTS_COUNTRY_REFERENCE_REQUEST_TIMEOUT_SECONDS",
        20.0,
    ),
    "ucdp_sync_enabled": ("EVENTS_UCDP_SYNC_ENABLED", True),
    "ucdp_sync_hours": ("EVENTS_UCDP_SYNC_HOURS", 24),
    "ucdp_lookback_years": ("EVENTS_UCDP_LOOKBACK_YEARS", 8),
    "ucdp_max_pages": ("EVENTS_UCDP_MAX_PAGES", 100),
    "ucdp_request_timeout_seconds": ("EVENTS_UCDP_REQUEST_TIMEOUT_SECONDS", 25.0),
    "mid_sync_enabled": ("EVENTS_MID_SYNC_ENABLED", True),
    "mid_sync_hours": ("EVENTS_MID_SYNC_HOURS", 168),
    "mid_request_timeout_seconds": ("EVENTS_MID_REQUEST_TIMEOUT_SECONDS", 20.0),
    "trade_dependency_sync_enabled": ("EVENTS_TRADE_DEPENDENCY_SYNC_ENABLED", True),
    "trade_dependency_sync_hours": ("EVENTS_TRADE_DEPENDENCY_SYNC_HOURS", 24),
    "trade_dependency_request_timeout_seconds": (
        "EVENTS_TRADE_DEPENDENCY_REQUEST_TIMEOUT_SECONDS",
        20.0,
    ),
    "trade_dependency_wb_per_page": ("EVENTS_TRADE_DEPENDENCY_WB_PER_PAGE", 5000),
    "trade_dependency_wb_max_pages": ("EVENTS_TRADE_DEPENDENCY_WB_MAX_PAGES", 50),
    "trade_dependency_base_divisor": ("EVENTS_TRADE_DEPENDENCY_BASE_DIVISOR", 120.0),
    "trade_dependency_min_factor": ("EVENTS_TRADE_DEPENDENCY_MIN_FACTOR", 0.5),
    "trade_dependency_max_factor": ("EVENTS_TRADE_DEPENDENCY_MAX_FACTOR", 1.5),
    "convergence_min_types": ("EVENTS_CONVERGENCE_MIN_TYPES", 2),
    "anomaly_threshold": ("EVENTS_ANOMALY_THRESHOLD", 1.8),
    "anomaly_min_baseline_points": ("EVENTS_ANOMALY_MIN_BASELINE_POINTS", 3),
    "instability_signal_min": ("EVENTS_INSTABILITY_SIGNAL_MIN", 15.0),
    "instability_critical": ("EVENTS_INSTABILITY_CRITICAL", 60.0),
    "tension_critical": ("EVENTS_TENSION_CRITICAL", 70.0),
    "gdelt_query_delay_seconds": ("EVENTS_GDELT_QUERY_DELAY_SECONDS", 5.0),
    "gdelt_max_concurrency": ("EVENTS_GDELT_MAX_CONCURRENCY", 1),
    "gdelt_news_enabled": ("EVENTS_GDELT_NEWS_ENABLED", True),
    "gdelt_news_timespan_hours": ("EVENTS_GDELT_NEWS_TIMESPAN_HOURS", 6),
    "gdelt_news_max_records": ("EVENTS_GDELT_NEWS_MAX_RECORDS", 40),
    "gdelt_news_request_timeout_seconds": (
        "EVENTS_GDELT_NEWS_REQUEST_TIMEOUT_SECONDS",
        20.0,
    ),
    "gdelt_news_cache_seconds": ("EVENTS_GDELT_NEWS_CACHE_SECONDS", 300),
    "gdelt_news_query_delay_seconds": ("EVENTS_GDELT_NEWS_QUERY_DELAY_SECONDS", 5.0),
    "gdelt_news_sync_enabled": ("EVENTS_GDELT_NEWS_SYNC_ENABLED", True),
    "gdelt_news_sync_hours": ("EVENTS_GDELT_NEWS_SYNC_HOURS", 1),
    "acled_rate_limit_per_min": ("EVENTS_ACLED_RATE_LIMIT_PER_MIN", 5),
    "acled_auth_rate_limit_per_min": ("EVENTS_ACLED_AUTH_RATE_LIMIT_PER_MIN", 12),
    "acled_cb_max_failures": ("EVENTS_ACLED_CB_MAX_FAILURES", 8),
    "acled_cb_cooldown_seconds": ("EVENTS_ACLED_CB_COOLDOWN_SECONDS", 180.0),
    "usgs_min_magnitude": ("EVENTS_USGS_MIN_MAGNITUDE", 4.5),
    "usgs_enabled": ("EVENTS_USGS_ENABLED", True),
}


def _coerce_setting(value: object, default: object) -> object:
    if value is None:
        return default
    if isinstance(default, bool):
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "on"}:
                return True
            if lowered in {"false", "0", "no", "off"}:
                return False
        return bool(value)
    if isinstance(default, int) and not isinstance(default, bool):
        try:
            return int(float(value))
        except Exception:
            return default
    if isinstance(default, float):
        try:
            return float(value)
        except Exception:
            return default
    if isinstance(default, list):
        return value if isinstance(value, list) else list(default)
    if isinstance(default, dict):
        return value if isinstance(value, dict) else dict(default)
    if isinstance(default, str):
        return str(value)
    return value


def _resolve_runtime_override(
    db_value: object,
    current_value: object,
    fallback_default: object,
) -> object:
    baseline = current_value if current_value is not None else fallback_default
    return _coerce_setting(db_value, baseline)


async def apply_events_settings():
    """Load events settings from DB into the runtime config singleton."""
    from models.database import AsyncSessionLocal, AppSettings
    from sqlalchemy import select
    from utils.secrets import decrypt_secret

    async with AsyncSessionLocal() as session:
        result = await session.execute(select(AppSettings).where(AppSettings.id == "default"))
        db = result.scalar_one_or_none()
        if not db:
            return

    raw_cfg = getattr(db, "events_settings_json", None)
    config_payload = raw_cfg if isinstance(raw_cfg, dict) else {}

    # Preserve older persisted GDELT-specific columns as fallback.
    if "gdelt_news_enabled" not in config_payload:
        config_payload["gdelt_news_enabled"] = getattr(db, "events_gdelt_news_enabled", None)
    if "gdelt_news_timespan_hours" not in config_payload:
        config_payload["gdelt_news_timespan_hours"] = getattr(db, "events_gdelt_news_timespan_hours", None)
    if "gdelt_news_max_records" not in config_payload:
        config_payload["gdelt_news_max_records"] = getattr(db, "events_gdelt_news_max_records", None)

    for db_field, (config_attr, default) in _EVENTS_DB_FIELD_MAP.items():
        current = getattr(settings, config_attr, default)
        resolved = _resolve_runtime_override(config_payload.get(db_field), current, default)
        object.__setattr__(settings, config_attr, resolved)

    current_acled_key = getattr(settings, "ACLED_API_KEY", None)
    current_acled_email = getattr(settings, "ACLED_EMAIL", None)
    current_opensky_user = getattr(settings, "OPENSKY_USERNAME", None)
    current_opensky_password = getattr(settings, "OPENSKY_PASSWORD", None)
    current_aisstream_key = getattr(settings, "AISSTREAM_API_KEY", None)
    current_cloudflare_token = getattr(settings, "CLOUDFLARE_RADAR_TOKEN", None)

    object.__setattr__(
        settings,
        "ACLED_API_KEY",
        _resolve_runtime_override(
            decrypt_secret(getattr(db, "events_acled_api_key", None)),
            current_acled_key,
            None,
        ),
    )
    object.__setattr__(
        settings,
        "ACLED_EMAIL",
        _resolve_runtime_override(
            str(getattr(db, "events_acled_email", "") or "").strip() or None,
            current_acled_email,
            None,
        ),
    )
    object.__setattr__(
        settings,
        "OPENSKY_USERNAME",
        _resolve_runtime_override(
            str(getattr(db, "events_opensky_username", "") or "").strip() or None,
            current_opensky_user,
            None,
        ),
    )
    object.__setattr__(
        settings,
        "OPENSKY_PASSWORD",
        _resolve_runtime_override(
            decrypt_secret(getattr(db, "events_opensky_password", None)),
            current_opensky_password,
            None,
        ),
    )
    object.__setattr__(
        settings,
        "AISSTREAM_API_KEY",
        _resolve_runtime_override(
            decrypt_secret(getattr(db, "events_aisstream_api_key", None)),
            current_aisstream_key,
            None,
        ),
    )
    object.__setattr__(
        settings,
        "CLOUDFLARE_RADAR_TOKEN",
        _resolve_runtime_override(
            decrypt_secret(getattr(db, "events_cloudflare_radar_token", None)),
            current_cloudflare_token,
            None,
        ),
    )


async def apply_search_filters():
    """Load search filter values from the DB and apply them to the running config singleton.

    Called at startup after DB init, and whenever search filter settings are updated via the API.
    """
    from models.database import AsyncSessionLocal, AppSettings
    from sqlalchemy import select

    async with AsyncSessionLocal() as session:
        result = await session.execute(select(AppSettings).where(AppSettings.id == "default"))
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
        ("BTC_ETH_HF_PURE_ARB_MAX_COMBINED", "btc_eth_pure_arb_max_combined", 0.98),
        ("BTC_ETH_HF_DUMP_THRESHOLD", "btc_eth_dump_hedge_drop_pct", 0.05),
        ("BTC_ETH_HF_THIN_LIQUIDITY_USD", "btc_eth_thin_liquidity_usd", 500.0),
        # Series IDs: single source of truth is the Settings class attribute.
        # Default=None here tells the apply loop to use the class default.
        ("BTC_ETH_HF_SERIES_BTC_15M", "btc_eth_hf_series_btc_15m", None),
        ("BTC_ETH_HF_SERIES_ETH_15M", "btc_eth_hf_series_eth_15m", None),
        ("BTC_ETH_HF_SERIES_SOL_15M", "btc_eth_hf_series_sol_15m", None),
        ("BTC_ETH_HF_SERIES_XRP_15M", "btc_eth_hf_series_xrp_15m", None),
        ("BTC_ETH_HF_SERIES_BTC_5M", "btc_eth_hf_series_btc_5m", None),
        ("BTC_ETH_HF_SERIES_ETH_5M", "btc_eth_hf_series_eth_5m", None),
        ("BTC_ETH_HF_SERIES_SOL_5M", "btc_eth_hf_series_sol_5m", None),
        ("BTC_ETH_HF_SERIES_XRP_5M", "btc_eth_hf_series_xrp_5m", None),
        ("BTC_ETH_HF_SERIES_BTC_1H", "btc_eth_hf_series_btc_1h", None),
        ("BTC_ETH_HF_SERIES_ETH_1H", "btc_eth_hf_series_eth_1h", None),
        ("BTC_ETH_HF_SERIES_SOL_1H", "btc_eth_hf_series_sol_1h", None),
        ("BTC_ETH_HF_SERIES_XRP_1H", "btc_eth_hf_series_xrp_1h", None),
        ("BTC_ETH_HF_SERIES_BTC_4H", "btc_eth_hf_series_btc_4h", None),
        ("BTC_ETH_HF_SERIES_ETH_4H", "btc_eth_hf_series_eth_4h", None),
        ("BTC_ETH_HF_SERIES_SOL_4H", "btc_eth_hf_series_sol_4h", None),
        ("BTC_ETH_HF_SERIES_XRP_4H", "btc_eth_hf_series_xrp_4h", None),
        # Risk scoring
        ("RISK_VERY_SHORT_DAYS", "risk_very_short_days", 2),
        ("RISK_SHORT_DAYS", "risk_short_days", 7),
        ("RISK_LONG_LOCKUP_DAYS", "risk_long_lockup_days", 180),
        ("RISK_EXTENDED_LOCKUP_DAYS", "risk_extended_lockup_days", 90),
        ("RISK_LOW_LIQUIDITY", "risk_low_liquidity", 1000.0),
        ("RISK_MODERATE_LIQUIDITY", "risk_moderate_liquidity", 5000.0),
        ("RISK_COMPLEX_LEGS", "risk_complex_legs", 5),
        ("RISK_MULTIPLE_LEGS", "risk_multiple_legs", 3),
        # BTC/ETH high-frequency strategy settings
        ("BTC_ETH_HF_ENABLED", "btc_eth_hf_enabled", True),
        ("BTC_ETH_HF_MAKER_MODE", "btc_eth_hf_maker_mode", True),
        # Scanner basics (already wired but also reloaded here for consistency)
        ("SCAN_INTERVAL_SECONDS", "scan_interval_seconds", 60),
        ("MIN_PROFIT_THRESHOLD", "min_profit_threshold", None),
        ("MAX_MARKETS_TO_SCAN", "max_markets_to_scan", 0),
        ("MAX_EVENTS_TO_SCAN", "max_events_to_scan", 0),
        ("MARKET_FETCH_PAGE_SIZE", "market_fetch_page_size", 200),
        ("MARKET_FETCH_ORDER", "market_fetch_order", "volume"),
        ("MIN_LIQUIDITY", "min_liquidity", 1000.0),
        ("SCANNER_MAX_OPPORTUNITIES_TOTAL", "scanner_max_opportunities_total", 500),
        ("SCANNER_MAX_OPPORTUNITIES_PER_STRATEGY", "scanner_max_opportunities_per_strategy", 120),
        # Trading safety limits (used by live_execution_service order validation + /trader-orchestrator/live/status)
        ("MAX_TRADE_SIZE_USD", "max_trade_size_usd", 100.0),
        ("MAX_DAILY_TRADE_VOLUME", "max_daily_trade_volume", 1000.0),
        ("MAX_SLIPPAGE_PERCENT", "max_slippage_percent", 2.0),
    ]

    for config_attr, db_attr, default in _apply:
        # When default is None, use the Settings class attribute as the single
        # source of truth (avoids duplicating literal values in two places).
        effective_default = default if default is not None else getattr(settings, config_attr, None)
        db_val = getattr(db, db_attr, None)
        if isinstance(db_val, str) and not str(db_val).strip() and effective_default is not None:
            db_val = effective_default
        if db_attr == "min_profit_threshold" and db_val is not None:
            db_val = db_val / 100.0

        current = getattr(settings, config_attr, effective_default)
        resolved = _resolve_runtime_override(db_val, current, effective_default)
        object.__setattr__(settings, config_attr, resolved)


async def apply_runtime_settings_overrides() -> None:
    """Apply DB runtime overrides with deterministic precedence.

    Precedence is always:
      1) non-null DB override
      2) environment value already loaded into ``settings``
      3) code default
    """
    await apply_events_settings()
    await apply_search_filters()
