from sqlalchemy import (
    Column,
    String,
    Integer,
    Boolean,
    DateTime,
    Text,
    JSON,
    ForeignKey,
    Enum as SQLEnum,
    Index,
    UniqueConstraint,
    event,
)
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
import enum
import logging
import os

from config import settings
from models.types import PreciseFloat as Float

logger = logging.getLogger(__name__)

Base = declarative_base()


class TradeStatus(enum.Enum):
    PENDING = "pending"
    OPEN = "open"
    RESOLVED_WIN = "resolved_win"
    RESOLVED_LOSS = "resolved_loss"
    CANCELLED = "cancelled"
    FAILED = "failed"


class PositionSide(enum.Enum):
    YES = "yes"
    NO = "no"


class CopyTradingMode(enum.Enum):
    ALL_TRADES = "all_trades"  # Mirror every trade from source wallet
    ARB_ONLY = "arb_only"  # Only copy trades matching detected arbitrage opportunities


# ==================== SIMULATION ACCOUNT ====================


class SimulationAccount(Base):
    """Simulated trading account for paper trading"""

    __tablename__ = "simulation_accounts"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    initial_capital = Column(Float, nullable=False, default=10000.0)
    current_capital = Column(Float, nullable=False, default=10000.0)
    total_pnl = Column(Float, nullable=False, default=0.0)
    total_trades = Column(Integer, nullable=False, default=0)
    winning_trades = Column(Integer, nullable=False, default=0)
    losing_trades = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Settings
    max_position_size_pct = Column(Float, default=10.0)  # Max % of capital per position
    max_open_positions = Column(Integer, default=10)
    slippage_model = Column(String, default="fixed")  # fixed, linear, sqrt
    slippage_bps = Column(Float, default=50.0)  # Basis points

    positions = relationship("SimulationPosition", back_populates="account")
    trades = relationship("SimulationTrade", back_populates="account")


class SimulationPosition(Base):
    """Open position in simulation account"""

    __tablename__ = "simulation_positions"

    id = Column(String, primary_key=True)
    account_id = Column(String, ForeignKey("simulation_accounts.id"), nullable=False)
    opportunity_id = Column(String, nullable=True)

    # Market details
    market_id = Column(String, nullable=False)
    market_question = Column(Text)
    token_id = Column(String)
    side = Column(SQLEnum(PositionSide), nullable=False)

    # Position details
    quantity = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    entry_cost = Column(Float, nullable=False)
    current_price = Column(Float, nullable=True)
    unrealized_pnl = Column(Float, default=0.0)

    # Risk management
    take_profit_price = Column(Float, nullable=True)
    stop_loss_price = Column(Float, nullable=True)

    # Timing
    opened_at = Column(DateTime, default=datetime.utcnow)
    resolution_date = Column(DateTime, nullable=True)

    # Status
    status = Column(SQLEnum(TradeStatus), default=TradeStatus.OPEN)

    account = relationship("SimulationAccount", back_populates="positions")

    __table_args__ = (
        Index("idx_position_account", "account_id"),
        Index("idx_position_market", "market_id"),
    )


class SimulationTrade(Base):
    """Completed trade in simulation account"""

    __tablename__ = "simulation_trades"

    id = Column(String, primary_key=True)
    account_id = Column(String, ForeignKey("simulation_accounts.id"), nullable=False)
    opportunity_id = Column(String, nullable=True)
    strategy_type = Column(String)

    # Execution details
    positions_data = Column(JSON)  # All positions taken
    total_cost = Column(Float, nullable=False)
    expected_profit = Column(Float)
    slippage = Column(Float, default=0.0)

    # Resolution
    status = Column(SQLEnum(TradeStatus), default=TradeStatus.PENDING)
    actual_payout = Column(Float, nullable=True)
    actual_pnl = Column(Float, nullable=True)
    fees_paid = Column(Float, default=0.0)

    # Timing
    executed_at = Column(DateTime, default=datetime.utcnow)
    resolved_at = Column(DateTime, nullable=True)

    # Copy trading reference
    copied_from_wallet = Column(String, nullable=True)

    account = relationship("SimulationAccount", back_populates="trades")

    __table_args__ = (
        Index("idx_trade_account", "account_id"),
        Index("idx_trade_status", "status"),
        Index("idx_trade_copied", "copied_from_wallet"),
    )


# ==================== COPY TRADING ====================


class CopyTradingConfig(Base):
    """Configuration for copy trading a wallet"""

    __tablename__ = "copy_trading_configs"

    id = Column(String, primary_key=True)
    source_wallet = Column(String, nullable=True, index=True)  # nullable for pool/tracked_group modes
    account_id = Column(String, ForeignKey("simulation_accounts.id"), nullable=False)
    source_type = Column(String, default="individual")  # individual | tracked_group | pool

    enabled = Column(Boolean, default=True)
    copy_mode = Column(SQLEnum(CopyTradingMode), default=CopyTradingMode.ALL_TRADES)
    min_roi_threshold = Column(
        Float, default=2.5
    )  # Only copy if ROI > X% (arb_only mode)
    max_position_size = Column(Float, default=1000.0)
    copy_delay_seconds = Column(Integer, default=5)
    slippage_tolerance = Column(Float, default=1.0)

    # Proportional sizing: scale positions relative to source wallet
    proportional_sizing = Column(Boolean, default=False)
    proportional_multiplier = Column(Float, default=1.0)  # 0.1 = 10% of source size

    # Trade direction control
    copy_buys = Column(Boolean, default=True)
    copy_sells = Column(Boolean, default=True)  # Mirror sell/close positions

    # Deduplication: track last processed trade timestamp
    last_processed_trade_id = Column(String, nullable=True)
    last_processed_timestamp = Column(DateTime, nullable=True)

    # Market filtering (JSON list of categories to include, empty = all)
    market_categories = Column(JSON, default=list)

    # Stats
    total_copied = Column(Integer, default=0)
    successful_copies = Column(Integer, default=0)
    failed_copies = Column(Integer, default=0)
    total_pnl = Column(Float, default=0.0)
    total_buys_copied = Column(Integer, default=0)
    total_sells_copied = Column(Integer, default=0)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (Index("idx_copy_wallet", "source_wallet"),)


class CopiedTrade(Base):
    """Record of a trade that was copied from a source wallet.
    Used for deduplication and tracking copy performance."""

    __tablename__ = "copied_trades"

    id = Column(String, primary_key=True)  # Our internal ID
    config_id = Column(String, ForeignKey("copy_trading_configs.id"), nullable=False)
    source_trade_id = Column(String, nullable=False)  # Trade ID from source wallet
    source_wallet = Column(String, nullable=False)

    # What was copied
    market_id = Column(String, nullable=False)
    market_question = Column(Text, nullable=True)
    token_id = Column(String, nullable=True)
    side = Column(String, nullable=False)  # BUY or SELL
    outcome = Column(String, nullable=True)  # YES or NO
    source_price = Column(Float, nullable=False)
    source_size = Column(Float, nullable=False)
    executed_price = Column(Float, nullable=True)  # Our actual execution price
    executed_size = Column(Float, nullable=True)  # Our actual size

    # Execution
    status = Column(String, default="pending")  # pending, executed, failed, skipped
    execution_mode = Column(String, default="simulation")  # simulation or live
    simulation_trade_id = Column(String, nullable=True)  # Links to SimulationTrade
    error_message = Column(Text, nullable=True)

    # Timing
    source_timestamp = Column(DateTime, nullable=True)
    copied_at = Column(DateTime, default=datetime.utcnow)
    executed_at = Column(DateTime, nullable=True)

    # PnL tracking
    realized_pnl = Column(Float, nullable=True)

    __table_args__ = (
        Index("idx_copied_config", "config_id"),
        Index("idx_copied_source_trade", "source_trade_id"),
        Index("idx_copied_source_wallet", "source_wallet"),
        Index("idx_copied_market", "market_id"),
        Index("idx_copied_status", "status"),
    )


# ==================== WALLET ANALYSIS ====================


class TrackedWallet(Base):
    """Wallet being tracked for analysis"""

    __tablename__ = "tracked_wallets"

    address = Column(String, primary_key=True)
    label = Column(String)
    added_at = Column(DateTime, default=datetime.utcnow)

    # Stats
    total_trades = Column(Integer, default=0)
    win_rate = Column(Float, nullable=True)
    total_pnl = Column(Float, default=0.0)
    avg_roi = Column(Float, nullable=True)
    last_trade_at = Column(DateTime, nullable=True)

    # Anomaly scores
    anomaly_score = Column(Float, default=0.0)
    is_flagged = Column(Boolean, default=False)
    flag_reasons = Column(JSON, default=list)

    # Analysis
    last_analyzed_at = Column(DateTime, nullable=True)
    analysis_data = Column(JSON, nullable=True)


class WalletTrade(Base):
    """Trade made by a tracked wallet"""

    __tablename__ = "wallet_trades"

    id = Column(String, primary_key=True)
    wallet_address = Column(
        String, ForeignKey("tracked_wallets.address"), nullable=False
    )

    # Trade details
    market_id = Column(String, nullable=False)
    market_question = Column(Text)
    side = Column(String)  # BUY/SELL
    outcome = Column(String)  # YES/NO
    price = Column(Float)
    amount = Column(Float)

    # Timing
    timestamp = Column(DateTime, nullable=False)
    block_number = Column(Integer, nullable=True)
    tx_hash = Column(String, nullable=True)

    # Analysis flags
    is_anomalous = Column(Boolean, default=False)
    anomaly_type = Column(String, nullable=True)
    anomaly_score = Column(Float, default=0.0)

    __table_args__ = (
        Index("idx_wallet_trade_wallet", "wallet_address"),
        Index("idx_wallet_trade_market", "market_id"),
        Index("idx_wallet_trade_time", "timestamp"),
    )


# ==================== OPPORTUNITIES ====================


class OpportunityHistory(Base):
    """Historical record of detected opportunities"""

    __tablename__ = "opportunity_history"

    id = Column(String, primary_key=True)
    strategy_type = Column(String, nullable=False)
    event_id = Column(String, nullable=True)

    # Opportunity details
    title = Column(Text)
    total_cost = Column(Float)
    expected_roi = Column(Float)
    risk_score = Column(Float)
    positions_data = Column(JSON)

    # Timing
    detected_at = Column(DateTime, default=datetime.utcnow)
    expired_at = Column(DateTime, nullable=True)
    resolution_date = Column(DateTime, nullable=True)

    # Outcome (if resolved)
    was_profitable = Column(Boolean, nullable=True)
    actual_roi = Column(Float, nullable=True)

    __table_args__ = (
        Index("idx_opp_strategy", "strategy_type"),
        Index("idx_opp_detected", "detected_at"),
    )


# ==================== OPPORTUNITY DECAY TRACKING ====================


class OpportunityLifetime(Base):
    """Tracks how long arbitrage opportunities survive before closing"""

    __tablename__ = "opportunity_lifetimes"

    id = Column(String, primary_key=True)
    opportunity_id = Column(String, nullable=False)
    strategy_type = Column(String, nullable=False)
    roi_at_detection = Column(Float, nullable=True)
    liquidity_at_detection = Column(Float, nullable=True)
    first_seen = Column(DateTime, nullable=False)
    last_seen = Column(DateTime, nullable=True)
    closed_at = Column(DateTime, nullable=True)
    lifetime_seconds = Column(Float, nullable=True)
    close_reason = Column(String, nullable=True)  # "price_moved", "resolved", "unknown"

    __table_args__ = (
        Index("idx_lifetime_strategy", "strategy_type"),
        Index("idx_lifetime_opportunity", "opportunity_id"),
        Index("idx_lifetime_first_seen", "first_seen"),
        Index("idx_lifetime_closed", "closed_at"),
    )


# ==================== NEWS INTELLIGENCE ====================


class NewsArticleCache(Base):
    """Persisted news article cache for matching/search."""

    __tablename__ = "news_article_cache"

    article_id = Column(String, primary_key=True)
    url = Column(Text, nullable=False)
    title = Column(Text, nullable=False)
    source = Column(String, nullable=True)
    feed_source = Column(String, nullable=True)
    category = Column(String, nullable=True)
    summary = Column(Text, nullable=True)
    published = Column(DateTime, nullable=True)
    fetched_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    embedding = Column(JSON, nullable=True)

    __table_args__ = (
        Index("idx_news_cache_fetched_at", "fetched_at"),
        Index("idx_news_cache_feed_source", "feed_source"),
        Index("idx_news_cache_category", "category"),
    )


class NewsMarketWatcher(Base):
    """Reverse index entry for a market watcher used by the news workflow."""

    __tablename__ = "news_market_watchers"

    market_id = Column(String, primary_key=True)
    question = Column(Text, nullable=False)
    event_title = Column(Text, nullable=True)
    category = Column(String, nullable=True)
    yes_price = Column(Float, nullable=True)
    no_price = Column(Float, nullable=True)
    liquidity = Column(Float, nullable=True)
    slug = Column(String, nullable=True)
    keywords = Column(JSON, nullable=True)
    embedding = Column(JSON, nullable=True)
    last_seen_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("idx_news_watcher_updated", "updated_at"),
        Index("idx_news_watcher_category", "category"),
        Index("idx_news_watcher_liquidity", "liquidity"),
    )


class NewsWorkflowFinding(Base):
    """Persisted result from the independent news workflow pipeline."""

    __tablename__ = "news_workflow_findings"

    id = Column(String, primary_key=True)
    article_id = Column(String, nullable=False, index=True)
    market_id = Column(String, nullable=False, index=True)
    article_title = Column(Text, nullable=False)
    article_source = Column(String, nullable=True)
    article_url = Column(Text, nullable=True)
    signal_key = Column(String, nullable=True, index=True)
    cache_key = Column(String, nullable=True, index=True)
    market_question = Column(Text, nullable=False)
    market_price = Column(Float, nullable=True)
    model_probability = Column(Float, nullable=True)
    edge_percent = Column(Float, nullable=True)
    direction = Column(String, nullable=True)
    confidence = Column(Float, nullable=True)
    retrieval_score = Column(Float, nullable=True)
    semantic_score = Column(Float, nullable=True)
    keyword_score = Column(Float, nullable=True)
    event_score = Column(Float, nullable=True)
    rerank_score = Column(Float, nullable=True)
    event_graph = Column(JSON, nullable=True)
    evidence = Column(JSON, nullable=True)
    reasoning = Column(Text, nullable=True)
    actionable = Column(Boolean, default=False, nullable=False)
    consumed_by_orchestrator = Column(Boolean, default=False, nullable=False)
    consumed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("idx_news_finding_created", "created_at"),
        Index("idx_news_finding_actionable", "actionable"),
        Index("idx_news_finding_consumed", "consumed_by_orchestrator"),
        Index("idx_news_finding_signal", "signal_key", unique=True),
    )


class NewsTradeIntent(Base):
    """Execution-oriented intent generated from high-conviction findings."""

    __tablename__ = "news_trade_intents"

    id = Column(String, primary_key=True)
    finding_id = Column(String, nullable=False, index=True)
    market_id = Column(String, nullable=False, index=True)
    market_question = Column(Text, nullable=False)
    direction = Column(String, nullable=False)  # buy_yes | buy_no
    signal_key = Column(String, nullable=True, index=True)
    entry_price = Column(Float, nullable=True)
    model_probability = Column(Float, nullable=True)
    edge_percent = Column(Float, nullable=True)
    confidence = Column(Float, nullable=True)
    suggested_size_usd = Column(Float, nullable=True)
    metadata_json = Column(JSON, nullable=True)
    status = Column(
        String, default="pending", nullable=False
    )  # pending | submitted | executed | skipped | expired
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    consumed_at = Column(DateTime, nullable=True)

    __table_args__ = (
        Index("idx_news_intent_created", "created_at"),
        Index("idx_news_intent_status", "status"),
        Index("idx_news_intent_market", "market_id"),
        Index("idx_news_intent_signal", "signal_key", unique=True),
    )


# ==================== ANOMALIES ====================


class DetectedAnomaly(Base):
    """Detected anomaly in trading data"""

    __tablename__ = "detected_anomalies"

    id = Column(String, primary_key=True)
    anomaly_type = Column(String, nullable=False)
    severity = Column(String, nullable=False)  # low, medium, high, critical

    # Subject
    wallet_address = Column(String, nullable=True)
    market_id = Column(String, nullable=True)
    trade_id = Column(String, nullable=True)

    # Details
    description = Column(Text)
    evidence = Column(JSON)
    score = Column(Float)

    # Timing
    detected_at = Column(DateTime, default=datetime.utcnow)

    # Resolution
    is_resolved = Column(Boolean, default=False)
    resolution_notes = Column(Text, nullable=True)

    __table_args__ = (
        Index("idx_anomaly_type", "anomaly_type"),
        Index("idx_anomaly_wallet", "wallet_address"),
        Index("idx_anomaly_severity", "severity"),
    )


# ==================== ML CLASSIFIER ====================


class MLModelWeights(Base):
    """Stored weights and metadata for the ML false-positive classifier"""

    __tablename__ = "ml_model_weights"

    id = Column(String, primary_key=True)
    model_version = Column(Integer, nullable=False, default=1)
    weights = Column(
        JSON, nullable=False
    )  # Model parameters (weights, bias, thresholds)
    feature_names = Column(JSON, nullable=False)  # Ordered list of feature names
    metrics = Column(JSON, nullable=True)  # accuracy, precision, recall, f1
    training_samples = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)


class MLPredictionLog(Base):
    """Log of ML classifier predictions for auditing and retraining"""

    __tablename__ = "ml_prediction_log"

    id = Column(String, primary_key=True)
    opportunity_id = Column(String, nullable=False)
    strategy_type = Column(String, nullable=False)
    features = Column(JSON, nullable=False)
    probability = Column(Float, nullable=False)
    recommendation = Column(String, nullable=False)  # execute, skip, review
    confidence = Column(Float, nullable=False)
    model_version = Column(Integer, nullable=True)
    predicted_at = Column(DateTime, default=datetime.utcnow)

    # Outcome tracking (filled in later)
    actual_outcome = Column(Boolean, nullable=True)
    actual_roi = Column(Float, nullable=True)

    __table_args__ = (
        Index("idx_ml_pred_opp", "opportunity_id"),
        Index("idx_ml_pred_time", "predicted_at"),
    )


# ==================== PARAMETER OPTIMIZATION ====================


class ParameterSet(Base):
    """Stored parameter sets for hyperparameter optimization"""

    __tablename__ = "parameter_sets"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    parameters = Column(JSON, nullable=False)
    backtest_results = Column(JSON, nullable=True)
    is_active = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class ValidationJob(Base):
    """Persistent async validation job queue (backtests/optimization)."""

    __tablename__ = "validation_jobs"

    id = Column(String, primary_key=True)
    job_type = Column(String, nullable=False)  # backtest | optimize
    status = Column(
        String, nullable=False, default="queued"
    )  # queued | running | completed | failed | cancelled
    payload = Column(JSON, nullable=True)
    result = Column(JSON, nullable=True)
    error = Column(Text, nullable=True)
    progress = Column(Float, default=0.0)
    message = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)

    __table_args__ = (
        Index("idx_validation_job_status", "status"),
        Index("idx_validation_job_created", "created_at"),
    )


class StrategyValidationProfile(Base):
    """Persisted health metrics and guardrail status per strategy."""

    __tablename__ = "strategy_validation_profiles"

    strategy_type = Column(String, primary_key=True)
    status = Column(String, nullable=False, default="active")  # active | demoted
    sample_size = Column(Integer, default=0)
    directional_accuracy = Column(Float, nullable=True)
    mae_roi = Column(Float, nullable=True)
    rmse_roi = Column(Float, nullable=True)
    optimism_bias_roi = Column(Float, nullable=True)
    last_reason = Column(Text, nullable=True)
    manual_override = Column(Boolean, default=False)
    manual_override_note = Column(String, nullable=True)
    demoted_at = Column(DateTime, nullable=True)
    restored_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("idx_validation_profile_status", "status"),
        Index("idx_validation_profile_updated", "updated_at"),
    )


# ==================== SCANNER SETTINGS ====================


class ScannerSettings(Base):
    """Persisted scanner configuration"""

    __tablename__ = "scanner_settings"

    id = Column(String, primary_key=True, default="default")
    is_enabled = Column(Boolean, default=True)
    scan_interval_seconds = Column(Integer, default=300)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# ==================== APP SETTINGS ====================


class AppSettings(Base):
    """Application-wide settings stored in database"""

    __tablename__ = "app_settings"

    id = Column(String, primary_key=True, default="default")

    # Polymarket Account Settings
    polymarket_api_key = Column(String, nullable=True)
    polymarket_api_secret = Column(String, nullable=True)
    polymarket_api_passphrase = Column(String, nullable=True)
    polymarket_private_key = Column(String, nullable=True)

    # Kalshi Account Settings
    kalshi_email = Column(String, nullable=True)
    kalshi_password = Column(String, nullable=True)
    kalshi_api_key = Column(String, nullable=True)

    # LLM/AI Service Settings
    openai_api_key = Column(String, nullable=True)
    anthropic_api_key = Column(String, nullable=True)
    llm_provider = Column(
        String, default="none"
    )  # none, openai, anthropic, google, xai, deepseek, ollama, lmstudio
    llm_model = Column(String, nullable=True)
    google_api_key = Column(String, nullable=True)
    xai_api_key = Column(String, nullable=True)
    deepseek_api_key = Column(String, nullable=True)
    ollama_api_key = Column(String, nullable=True)
    ollama_base_url = Column(String, nullable=True)
    lmstudio_api_key = Column(String, nullable=True)
    lmstudio_base_url = Column(String, nullable=True)

    # AI Feature Settings
    ai_enabled = Column(Boolean, default=False)  # Master switch for AI features
    ai_resolution_analysis = Column(
        Boolean, default=True
    )  # Auto-analyze resolution criteria
    ai_opportunity_scoring = Column(Boolean, default=True)  # LLM-as-judge scoring
    ai_news_sentiment = Column(Boolean, default=True)  # News/sentiment analysis
    ai_max_monthly_spend = Column(Float, default=50.0)  # Monthly LLM cost cap
    ai_default_model = Column(
        String, default="gpt-4o-mini"
    )  # Default model for AI tasks
    ai_premium_model = Column(String, default="gpt-4o")  # Model for high-value analysis

    # Notification Settings
    telegram_bot_token = Column(String, nullable=True)
    telegram_chat_id = Column(String, nullable=True)
    notifications_enabled = Column(Boolean, default=False)
    notify_on_opportunity = Column(Boolean, default=True)
    notify_on_trade = Column(Boolean, default=True)
    notify_min_roi = Column(Float, default=5.0)
    notify_autotrader_orders = Column(Boolean, default=False)
    notify_autotrader_issues = Column(Boolean, default=True)
    notify_autotrader_timeline = Column(Boolean, default=True)
    notify_autotrader_summary_interval_minutes = Column(Integer, default=60)
    notify_autotrader_summary_per_trader = Column(Boolean, default=False)

    # Scanner Settings
    scan_interval_seconds = Column(Integer, default=60)
    min_profit_threshold = Column(Float, default=2.5)
    max_markets_to_scan = Column(Integer, default=500)
    min_liquidity = Column(Float, default=1000.0)

    # Discovery Engine Settings
    discovery_max_discovered_wallets = Column(Integer, default=20_000)
    discovery_maintenance_enabled = Column(Boolean, default=True)
    discovery_keep_recent_trade_days = Column(Integer, default=7)
    discovery_keep_new_discoveries_days = Column(Integer, default=30)
    discovery_maintenance_batch = Column(Integer, default=900)
    discovery_stale_analysis_hours = Column(Integer, default=12)
    discovery_analysis_priority_batch_limit = Column(Integer, default=2500)
    discovery_delay_between_markets = Column(Float, default=0.25)
    discovery_delay_between_wallets = Column(Float, default=0.15)
    discovery_max_markets_per_run = Column(Integer, default=100)
    discovery_max_wallets_per_market = Column(Integer, default=50)
    # Opportunities -> Traders UI defaults (persisted user preferences)
    discovery_trader_opps_source_filter = Column(String, default="all")
    discovery_trader_opps_min_tier = Column(String, default="WATCH")
    discovery_trader_opps_side_filter = Column(String, default="all")
    discovery_trader_opps_confluence_limit = Column(Integer, default=50)
    discovery_trader_opps_insider_limit = Column(Integer, default=40)
    discovery_trader_opps_insider_min_confidence = Column(Float, default=0.62)
    discovery_trader_opps_insider_max_age_minutes = Column(Integer, default=180)

    # Trading Safety Settings
    trading_enabled = Column(Boolean, default=False)
    max_trade_size_usd = Column(Float, default=100.0)
    max_daily_trade_volume = Column(Float, default=1000.0)
    max_open_positions = Column(Integer, default=10)
    max_slippage_percent = Column(Float, default=2.0)

    # Opportunity Search Filters (hard rejection thresholds)
    min_liquidity_hard = Column(Float, default=200.0)
    min_position_size = Column(Float, default=25.0)
    min_absolute_profit = Column(Float, default=5.0)
    min_annualized_roi = Column(Float, default=10.0)
    max_resolution_months = Column(Integer, default=18)
    max_plausible_roi = Column(Float, default=30.0)
    max_trade_legs = Column(Integer, default=8)
    min_liquidity_per_leg = Column(Float, default=500.0)

    # NegRisk Exhaustivity Thresholds
    negrisk_min_total_yes = Column(Float, default=0.95)
    negrisk_warn_total_yes = Column(Float, default=0.97)
    negrisk_election_min_total_yes = Column(Float, default=0.97)
    negrisk_max_resolution_spread_days = Column(Integer, default=7)

    # Settlement Lag
    settlement_lag_max_days_to_resolution = Column(Integer, default=14)
    settlement_lag_near_zero = Column(Float, default=0.05)
    settlement_lag_near_one = Column(Float, default=0.95)
    settlement_lag_min_sum_deviation = Column(Float, default=0.03)

    # Risk Scoring Thresholds
    risk_very_short_days = Column(Integer, default=2)
    risk_short_days = Column(Integer, default=7)
    risk_long_lockup_days = Column(Integer, default=180)
    risk_extended_lockup_days = Column(Integer, default=90)
    risk_low_liquidity = Column(Float, default=1000.0)
    risk_moderate_liquidity = Column(Float, default=5000.0)
    risk_complex_legs = Column(Integer, default=5)
    risk_multiple_legs = Column(Integer, default=3)

    # BTC/ETH High-Frequency Strategy
    btc_eth_pure_arb_max_combined = Column(Float, default=0.98)
    btc_eth_dump_hedge_drop_pct = Column(Float, default=0.05)
    btc_eth_thin_liquidity_usd = Column(Float, default=500.0)
    # Polymarket series IDs for crypto up-or-down markets
    btc_eth_hf_series_btc_15m = Column(String, default="10192")
    btc_eth_hf_series_eth_15m = Column(String, default="10191")
    btc_eth_hf_series_sol_15m = Column(String, default="10423")
    btc_eth_hf_series_xrp_15m = Column(String, default="10422")
    btc_eth_hf_series_btc_5m = Column(String, default="")
    btc_eth_hf_series_eth_5m = Column(String, default="")
    btc_eth_hf_series_sol_5m = Column(String, default="")
    btc_eth_hf_series_xrp_5m = Column(String, default="")
    btc_eth_hf_series_btc_1h = Column(String, default="10114")
    btc_eth_hf_series_eth_1h = Column(String, default="10117")
    btc_eth_hf_series_sol_1h = Column(String, default="10122")
    btc_eth_hf_series_xrp_1h = Column(String, default="10123")
    btc_eth_hf_series_btc_4h = Column(String, default="10331")
    btc_eth_hf_series_eth_4h = Column(String, default="10332")
    btc_eth_hf_series_sol_4h = Column(String, default="10326")
    btc_eth_hf_series_xrp_4h = Column(String, default="10327")

    # Miracle Strategy
    miracle_min_no_price = Column(Float, default=0.90)
    miracle_max_no_price = Column(Float, default=0.995)
    miracle_min_impossibility_score = Column(Float, default=0.70)

    # BTC/ETH High-Frequency Enable
    btc_eth_hf_enabled = Column(Boolean, default=True)
    btc_eth_hf_maker_mode = Column(Boolean, default=True)

    # Cross-Platform Arbitrage
    cross_platform_enabled = Column(Boolean, default=True)

    # Combinatorial Arbitrage
    combinatorial_min_confidence = Column(Float, default=0.75)
    combinatorial_high_confidence = Column(Float, default=0.90)

    # Bayesian Cascade
    bayesian_cascade_enabled = Column(Boolean, default=True)
    bayesian_min_edge_percent = Column(Float, default=5.0)
    bayesian_propagation_depth = Column(Integer, default=3)

    # Liquidity Vacuum
    liquidity_vacuum_enabled = Column(Boolean, default=True)
    liquidity_vacuum_min_imbalance_ratio = Column(Float, default=5.0)
    liquidity_vacuum_min_depth_usd = Column(Float, default=100.0)

    # Entropy Arbitrage
    entropy_arb_enabled = Column(Boolean, default=True)
    entropy_arb_min_deviation = Column(Float, default=0.25)

    # Event-Driven Arbitrage
    event_driven_enabled = Column(Boolean, default=True)

    # Temporal Decay
    temporal_decay_enabled = Column(Boolean, default=True)

    # Correlation Arbitrage
    correlation_arb_enabled = Column(Boolean, default=True)
    correlation_arb_min_correlation = Column(Float, default=0.7)
    correlation_arb_min_divergence = Column(Float, default=0.05)

    # Market Making
    market_making_enabled = Column(Boolean, default=True)
    market_making_spread_bps = Column(Float, default=100.0)
    market_making_max_inventory_usd = Column(Float, default=500.0)

    # Statistical Arbitrage
    stat_arb_enabled = Column(Boolean, default=True)
    stat_arb_min_edge = Column(Float, default=0.05)

    # Database Maintenance
    auto_cleanup_enabled = Column(Boolean, default=False)
    cleanup_interval_hours = Column(Integer, default=24)
    cleanup_resolved_trade_days = Column(Integer, default=30)
    market_cache_hygiene_enabled = Column(Boolean, default=True)
    market_cache_hygiene_interval_hours = Column(Integer, default=6)
    market_cache_retention_days = Column(Integer, default=120)
    market_cache_reference_lookback_days = Column(Integer, default=45)
    market_cache_weak_entry_grace_days = Column(Integer, default=7)
    market_cache_max_entries_per_slug = Column(Integer, default=3)

    # Trading VPN/Proxy (routes ONLY trading requests through proxy)
    trading_proxy_enabled = Column(Boolean, default=False)
    trading_proxy_url = Column(
        String, nullable=True
    )  # socks5://host:port, http://host:port
    trading_proxy_verify_ssl = Column(Boolean, default=True)
    trading_proxy_timeout = Column(Float, default=30.0)
    trading_proxy_require_vpn = Column(
        Boolean, default=True
    )  # Block trades if VPN unreachable

    # Validation guardrails (auto strategy demotion/promotion)
    validation_guardrails_enabled = Column(Boolean, default=True)
    validation_min_samples = Column(Integer, default=25)
    validation_min_directional_accuracy = Column(Float, default=0.52)
    validation_max_mae_roi = Column(Float, default=12.0)
    validation_lookback_days = Column(Integer, default=90)
    validation_auto_promote = Column(Boolean, default=True)

    # Independent News Workflow (Option B/C/D pipeline)
    news_workflow_enabled = Column(Boolean, default=True)
    news_workflow_auto_run = Column(Boolean, default=True)
    news_workflow_top_k = Column(Integer, default=20)
    news_workflow_rerank_top_n = Column(Integer, default=8)
    news_workflow_similarity_threshold = Column(Float, default=0.20)
    news_workflow_keyword_weight = Column(Float, default=0.25)
    news_workflow_semantic_weight = Column(Float, default=0.45)
    news_workflow_event_weight = Column(Float, default=0.30)
    news_workflow_require_verifier = Column(Boolean, default=True)
    news_workflow_market_min_liquidity = Column(Float, default=500.0)
    news_workflow_market_max_days_to_resolution = Column(Integer, default=365)
    news_workflow_min_keyword_signal = Column(Float, default=0.04)
    news_workflow_min_semantic_signal = Column(Float, default=0.05)
    news_workflow_min_edge_percent = Column(Float, default=8.0)
    news_workflow_min_confidence = Column(Float, default=0.6)
    news_workflow_require_second_source = Column(Boolean, default=False)
    news_workflow_orchestrator_enabled = Column(Boolean, default=True)
    news_workflow_orchestrator_min_edge = Column(Float, default=10.0)
    news_workflow_orchestrator_max_age_minutes = Column(Integer, default=120)
    news_workflow_scan_interval_seconds = Column(Integer, default=120)
    news_workflow_model = Column(String, nullable=True)
    news_workflow_cycle_spend_cap_usd = Column(Float, default=0.25)
    news_workflow_hourly_spend_cap_usd = Column(Float, default=2.0)
    news_workflow_cycle_llm_call_cap = Column(Integer, default=30)
    news_workflow_cache_ttl_minutes = Column(Integer, default=30)
    news_workflow_max_edge_evals_per_article = Column(Integer, default=6)
    news_rss_feeds_json = Column(JSON, default=list)
    news_gov_rss_enabled = Column(Boolean, default=True)
    news_gov_rss_feeds_json = Column(JSON, default=list)
    world_intel_settings_json = Column(JSON, default=dict)
    world_intel_acled_api_key = Column(String, nullable=True)
    world_intel_acled_email = Column(String, nullable=True)
    world_intel_opensky_username = Column(String, nullable=True)
    world_intel_opensky_password = Column(String, nullable=True)
    world_intel_aisstream_api_key = Column(String, nullable=True)
    world_intel_cloudflare_radar_token = Column(String, nullable=True)
    world_intel_country_reference_json = Column(JSON, default=list)
    world_intel_country_reference_source = Column(String, nullable=True)
    world_intel_country_reference_synced_at = Column(DateTime, nullable=True)
    world_intel_ucdp_active_wars_json = Column(JSON, default=list)
    world_intel_ucdp_minor_conflicts_json = Column(JSON, default=list)
    world_intel_ucdp_source = Column(String, nullable=True)
    world_intel_ucdp_year = Column(Integer, nullable=True)
    world_intel_ucdp_synced_at = Column(DateTime, nullable=True)
    world_intel_mid_iso3_json = Column(JSON, default=dict)
    world_intel_mid_source = Column(String, nullable=True)
    world_intel_mid_synced_at = Column(DateTime, nullable=True)
    world_intel_trade_dependencies_json = Column(JSON, default=dict)
    world_intel_trade_dependency_source = Column(String, nullable=True)
    world_intel_trade_dependency_year = Column(Integer, nullable=True)
    world_intel_trade_dependency_synced_at = Column(DateTime, nullable=True)
    world_intel_chokepoints_json = Column(JSON, default=list)
    world_intel_chokepoints_source = Column(String, nullable=True)
    world_intel_chokepoints_synced_at = Column(DateTime, nullable=True)
    world_intel_gdelt_news_enabled = Column(Boolean, default=True)
    world_intel_gdelt_news_queries_json = Column(JSON, default=list)
    world_intel_gdelt_news_timespan_hours = Column(Integer, default=6)
    world_intel_gdelt_news_max_records = Column(Integer, default=40)
    world_intel_gdelt_news_source = Column(String, nullable=True)
    world_intel_gdelt_news_synced_at = Column(DateTime, nullable=True)

    # Independent Weather Workflow (forecast consensus -> opportunities/intents)
    weather_workflow_enabled = Column(Boolean, default=True)
    weather_workflow_auto_run = Column(Boolean, default=True)
    weather_workflow_scan_interval_seconds = Column(Integer, default=14400)
    weather_workflow_entry_max_price = Column(Float, default=0.25)
    weather_workflow_take_profit_price = Column(Float, default=0.85)
    weather_workflow_stop_loss_pct = Column(Float, default=50.0)
    weather_workflow_min_edge_percent = Column(Float, default=8.0)
    weather_workflow_min_confidence = Column(Float, default=0.6)
    weather_workflow_min_model_agreement = Column(Float, default=0.75)
    weather_workflow_min_liquidity = Column(Float, default=500.0)
    weather_workflow_max_markets_per_scan = Column(Integer, default=200)
    weather_workflow_orchestrator_enabled = Column(Boolean, default=True)
    weather_workflow_orchestrator_min_edge = Column(Float, default=10.0)
    weather_workflow_orchestrator_max_age_minutes = Column(Integer, default=240)
    weather_workflow_default_size_usd = Column(Float, default=10.0)
    weather_workflow_max_size_usd = Column(Float, default=50.0)
    weather_workflow_model = Column(String, nullable=True)
    weather_workflow_temperature_unit = Column(String, default="F")

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# ==================== STRATEGY PLUGINS ====================


class StrategyPlugin(Base):
    """User-defined strategy plugins — full Python strategy implementations.

    Each plugin is a complete strategy file defining a BaseStrategy subclass
    with its own detect() method. Plugins run alongside built-in strategies
    during each scan cycle, receiving the same events/markets/prices data.
    """

    __tablename__ = "strategy_plugins"

    id = Column(String, primary_key=True)  # UUID
    slug = Column(String, unique=True, nullable=False)  # Unique identifier e.g. "whale_follower"
    name = Column(String, nullable=False)  # Display name (extracted from class or user-set)
    description = Column(Text, nullable=True)  # Strategy description
    source_code = Column(Text, nullable=False)  # Full Python source code
    class_name = Column(String, nullable=True)  # Extracted strategy class name
    enabled = Column(Boolean, default=True)
    status = Column(String, default="unloaded")  # unloaded, loaded, error
    error_message = Column(Text, nullable=True)  # Last load/validation error
    config = Column(JSON, default=dict)  # User config overrides passed to plugin
    version = Column(Integer, default=1)  # Bumped on each code edit
    sort_order = Column(Integer, default=0)  # Display order
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("idx_strategy_plugin_enabled", "enabled"),
        Index("idx_strategy_plugin_slug", "slug"),
    )


# ==================== LLM MODELS CACHE ====================


class LLMModelCache(Base):
    """Cached list of available models from each LLM provider.

    Models are fetched from provider APIs and stored here for quick
    lookup in the UI dropdown. Can be refreshed on demand.
    """

    __tablename__ = "llm_model_cache"

    id = Column(String, primary_key=True)
    provider = Column(
        String, nullable=False
    )  # openai, anthropic, google, xai, deepseek, ollama, lmstudio
    model_id = Column(String, nullable=False)  # The model identifier used in API calls
    display_name = Column(String, nullable=True)  # Human-readable name
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_llm_model_provider", "provider"),
        Index("idx_llm_model_id", "provider", "model_id", unique=True),
    )


# ==================== AI INTELLIGENCE LAYER ====================


class ResearchSession(Base):
    """Tracks a complete AI research session (e.g., one resolution analysis run).

    Each session represents a single research task executed by the AI system,
    including all LLM calls, tool invocations, and the final result.
    """

    __tablename__ = "research_sessions"

    id = Column(String, primary_key=True)
    session_type = Column(
        String, nullable=False
    )  # "resolution_analysis", "opportunity_judge", "market_analysis", "news_sentiment"
    query = Column(Text, nullable=False)  # The question/task being researched
    opportunity_id = Column(String, nullable=True)  # Link to opportunity if applicable
    market_id = Column(String, nullable=True)

    # Status
    status = Column(String, default="running")  # running, completed, failed, timeout
    result = Column(JSON, nullable=True)  # Final structured result
    error = Column(Text, nullable=True)

    # Agent metrics
    iterations = Column(Integer, default=0)
    tools_called = Column(Integer, default=0)

    # Token usage
    total_input_tokens = Column(Integer, default=0)
    total_output_tokens = Column(Integer, default=0)
    total_cost_usd = Column(Float, default=0.0)
    model_used = Column(String, nullable=True)

    # Timing
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    duration_seconds = Column(Float, nullable=True)

    entries = relationship(
        "ScratchpadEntry", back_populates="session", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_research_type", "session_type"),
        Index("idx_research_opp", "opportunity_id"),
        Index("idx_research_market", "market_id"),
        Index("idx_research_started", "started_at"),
    )


class ScratchpadEntry(Base):
    """Individual step in a research session.

    Replaces Dexter's JSONL scratchpad with a structured database table.
    Each entry represents a single thinking step, tool call, or observation
    within a research session.
    """

    __tablename__ = "scratchpad_entries"

    id = Column(String, primary_key=True)
    session_id = Column(String, ForeignKey("research_sessions.id"), nullable=False)
    sequence = Column(Integer, nullable=False)  # Order within session

    # Entry content
    entry_type = Column(
        String, nullable=False
    )  # "thinking", "tool_call", "tool_result", "observation", "answer"
    tool_name = Column(String, nullable=True)  # Which tool was called
    input_data = Column(JSON, nullable=True)  # Tool input or thinking content
    output_data = Column(JSON, nullable=True)  # Tool output or result

    # Token tracking
    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)

    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("ResearchSession", back_populates="entries")

    __table_args__ = (
        Index("idx_scratchpad_session", "session_id"),
        Index("idx_scratchpad_type", "entry_type"),
    )


class AIChatSession(Base):
    """Persistent copilot chat session."""

    __tablename__ = "ai_chat_sessions"

    id = Column(String, primary_key=True)
    context_type = Column(String, nullable=True)  # opportunity | market | general
    context_id = Column(String, nullable=True)
    title = Column(String, nullable=True)
    archived = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("idx_ai_chat_context", "context_type", "context_id"),
        Index("idx_ai_chat_updated", "updated_at"),
        Index("idx_ai_chat_archived", "archived"),
    )


class AIChatMessage(Base):
    """Message row for a persistent copilot chat session."""

    __tablename__ = "ai_chat_messages"

    id = Column(String, primary_key=True)
    session_id = Column(
        String,
        ForeignKey("ai_chat_sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    role = Column(String, nullable=False)  # system | user | assistant
    content = Column(Text, nullable=False)
    model_used = Column(String, nullable=True)
    input_tokens = Column(Integer, default=0, nullable=False)
    output_tokens = Column(Integer, default=0, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("idx_ai_chat_msg_session", "session_id"),
        Index("idx_ai_chat_msg_created", "created_at"),
    )


class ResolutionAnalysis(Base):
    """Cached resolution criteria analysis for a market.

    Stores LLM-generated analysis of a market's resolution rules,
    including clarity scores, identified ambiguities, edge cases,
    and a recommendation on whether to trade the market.
    """

    __tablename__ = "resolution_analyses"

    id = Column(String, primary_key=True)
    market_id = Column(String, nullable=False, index=True)
    condition_id = Column(String, nullable=True)

    # Market info
    question = Column(Text, nullable=False)
    resolution_source = Column(Text, nullable=True)
    resolution_rules = Column(Text, nullable=True)

    # Analysis results
    clarity_score = Column(
        Float, nullable=True
    )  # 0-1: how clear/unambiguous the resolution criteria are
    risk_score = Column(Float, nullable=True)  # 0-1: risk of unexpected resolution
    confidence = Column(Float, nullable=True)  # 0-1: confidence in the analysis

    # Detailed findings
    ambiguities = Column(JSON, nullable=True)  # List of identified ambiguities
    edge_cases = Column(JSON, nullable=True)  # Potential edge cases
    key_dates = Column(JSON, nullable=True)  # Important dates for resolution
    resolution_likelihood = Column(
        JSON, nullable=True
    )  # Likelihood assessment per outcome
    summary = Column(Text, nullable=True)  # Human-readable summary
    recommendation = Column(String, nullable=True)  # "safe", "caution", "avoid"

    # Metadata
    session_id = Column(String, ForeignKey("research_sessions.id"), nullable=True)
    model_used = Column(String, nullable=True)
    analyzed_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)  # When to re-analyze

    __table_args__ = (
        Index("idx_resolution_market", "market_id"),
        Index("idx_resolution_analyzed", "analyzed_at"),
    )


class OpportunityJudgment(Base):
    """LLM-as-judge scores for arbitrage opportunities.

    Stores multi-dimensional scoring from the LLM judge, including
    profit viability, resolution safety, execution feasibility,
    and comparison with the ML classifier's assessment.
    """

    __tablename__ = "opportunity_judgments"

    id = Column(String, primary_key=True)
    opportunity_id = Column(String, nullable=False)
    strategy_type = Column(String, nullable=False)

    # Scores (0.0 to 1.0)
    overall_score = Column(Float, nullable=False)  # Composite score
    profit_viability = Column(Float, nullable=True)  # Will the profit materialize?
    resolution_safety = Column(Float, nullable=True)  # Will it resolve as expected?
    execution_feasibility = Column(
        Float, nullable=True
    )  # Can we execute at these prices?
    market_efficiency = Column(
        Float, nullable=True
    )  # Is this a real inefficiency or noise?

    # LLM reasoning
    reasoning = Column(Text, nullable=True)  # Concise decision rationale
    recommendation = Column(
        String, nullable=False
    )  # "strong_execute", "execute", "review", "skip", "strong_skip"
    risk_factors = Column(JSON, nullable=True)

    # Comparison with ML classifier
    ml_probability = Column(Float, nullable=True)  # ML classifier's probability
    ml_recommendation = Column(String, nullable=True)  # ML classifier's recommendation
    agreement = Column(Boolean, nullable=True)  # Do ML and LLM agree?

    # Metadata
    session_id = Column(String, ForeignKey("research_sessions.id"), nullable=True)
    model_used = Column(String, nullable=True)
    judged_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_judgment_opp", "opportunity_id"),
        Index("idx_judgment_strategy", "strategy_type"),
        Index("idx_judgment_score", "overall_score"),
    )


class SkillExecution(Base):
    """Tracks individual skill executions within the AI system.

    Skills are reusable analysis workflows (e.g., resolution analysis,
    news lookup) that can be composed into larger research sessions.
    """

    __tablename__ = "skill_executions"

    id = Column(String, primary_key=True)
    skill_name = Column(String, nullable=False)
    session_id = Column(String, ForeignKey("research_sessions.id"), nullable=True)

    # Input/output
    input_context = Column(JSON, nullable=True)
    output_result = Column(JSON, nullable=True)

    # Status
    status = Column(String, default="running")  # running, completed, failed
    error = Column(Text, nullable=True)

    # Timing
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    duration_seconds = Column(Float, nullable=True)

    __table_args__ = (
        Index("idx_skill_name", "skill_name"),
        Index("idx_skill_session", "session_id"),
    )


class LLMUsageLog(Base):
    """Tracks LLM API usage for cost management and observability.

    Every LLM API call is logged here with token counts, costs,
    latency, and error information. Used for spend tracking,
    rate limiting, and debugging.
    """

    __tablename__ = "llm_usage_log"

    id = Column(String, primary_key=True)
    provider = Column(
        String, nullable=False
    )  # openai, anthropic, google, xai, deepseek, ollama, lmstudio
    model = Column(String, nullable=False)

    # Usage
    input_tokens = Column(Integer, nullable=False)
    output_tokens = Column(Integer, nullable=False)
    cost_usd = Column(Float, nullable=False)

    # Context
    purpose = Column(
        String, nullable=True
    )  # "resolution_analysis", "opportunity_judge", etc.
    session_id = Column(String, nullable=True)

    # Timing
    requested_at = Column(DateTime, default=datetime.utcnow)
    latency_ms = Column(Integer, nullable=True)

    # Error tracking
    success = Column(Boolean, default=True)
    error = Column(Text, nullable=True)

    __table_args__ = (
        Index("idx_llm_usage_provider", "provider"),
        Index("idx_llm_usage_model", "model"),
        Index("idx_llm_usage_time", "requested_at"),
        Index("idx_llm_usage_time_success", "requested_at", "success"),
        Index("idx_llm_usage_purpose", "purpose"),
    )


# ==================== TRADER DISCOVERY ====================


class DiscoveredWallet(Base):
    """Wallet discovered and profiled by the automated discovery engine.
    Contains comprehensive performance metrics, risk-adjusted scores, and rolling window stats."""

    __tablename__ = "discovered_wallets"

    address = Column(String, primary_key=True)
    username = Column(String, nullable=True)  # Polymarket username if resolved

    # Discovery metadata
    discovered_at = Column(DateTime, default=datetime.utcnow)
    last_analyzed_at = Column(DateTime, nullable=True)
    discovery_source = Column(String, default="scan")  # scan, manual, referral

    # Basic stats
    total_trades = Column(Integer, default=0)
    wins = Column(Integer, default=0)
    losses = Column(Integer, default=0)
    win_rate = Column(Float, default=0.0)
    total_pnl = Column(Float, default=0.0)
    realized_pnl = Column(Float, default=0.0)
    unrealized_pnl = Column(Float, default=0.0)
    total_invested = Column(Float, default=0.0)
    total_returned = Column(Float, default=0.0)
    avg_roi = Column(Float, default=0.0)
    max_roi = Column(Float, default=0.0)
    min_roi = Column(Float, default=0.0)
    roi_std = Column(Float, default=0.0)
    unique_markets = Column(Integer, default=0)
    open_positions = Column(Integer, default=0)
    days_active = Column(Integer, default=0)
    avg_hold_time_hours = Column(Float, default=0.0)
    trades_per_day = Column(Float, default=0.0)
    avg_position_size = Column(Float, default=0.0)

    # Risk-adjusted metrics
    sharpe_ratio = Column(Float, nullable=True)
    sortino_ratio = Column(Float, nullable=True)
    max_drawdown = Column(
        Float, nullable=True
    )  # Stored as positive fraction (0.15 = 15% drawdown)
    profit_factor = Column(Float, nullable=True)  # gross_profit / gross_loss
    calmar_ratio = Column(Float, nullable=True)  # annualized_return / max_drawdown

    # Rolling window metrics (JSON dicts keyed by period: "1d", "7d", "30d", "90d")
    rolling_pnl = Column(JSON, nullable=True)  # {"1d": 50.0, "7d": 200.0, ...}
    rolling_roi = Column(JSON, nullable=True)
    rolling_win_rate = Column(JSON, nullable=True)
    rolling_trade_count = Column(JSON, nullable=True)
    rolling_sharpe = Column(JSON, nullable=True)

    # Classification
    anomaly_score = Column(Float, default=0.0)
    is_bot = Column(Boolean, default=False)
    is_profitable = Column(Boolean, default=False)
    recommendation = Column(
        String, default="unanalyzed"
    )  # copy_candidate, monitor, avoid, unanalyzed
    strategies_detected = Column(JSON, default=list)

    # Leaderboard ranking (computed periodically)
    rank_score = Column(Float, default=0.0)  # Composite score for sorting
    rank_position = Column(Integer, nullable=True)  # Position on leaderboard
    metrics_source_version = Column(String, nullable=True)

    # Smart pool scoring (quality + recency + stability blend)
    quality_score = Column(Float, default=0.0)
    activity_score = Column(Float, default=0.0)
    stability_score = Column(Float, default=0.0)
    composite_score = Column(Float, default=0.0)

    # Near-real-time activity metrics
    last_trade_at = Column(DateTime, nullable=True)
    trades_1h = Column(Integer, default=0)
    trades_24h = Column(Integer, default=0)
    unique_markets_24h = Column(Integer, default=0)

    # Smart wallet pool membership
    in_top_pool = Column(Boolean, default=False)
    pool_tier = Column(String, nullable=True)  # core, rising, standby
    pool_membership_reason = Column(String, nullable=True)
    source_flags = Column(JSON, default=dict)  # {"leaderboard": true, ...}

    # Tags (many-to-many via JSON for simplicity in SQLite)
    tags = Column(JSON, default=list)  # ["smart_predictor", "whale", "consistent", ...]

    # Entity clustering
    cluster_id = Column(String, nullable=True)  # Which cluster this wallet belongs to

    # Insider detection (balanced mode)
    insider_score = Column(Float, default=0.0)
    insider_confidence = Column(Float, default=0.0)
    insider_sample_size = Column(Integer, default=0)
    insider_last_scored_at = Column(DateTime, nullable=True)
    insider_metrics_json = Column(JSON, nullable=True)
    insider_reasons_json = Column(JSON, default=list)

    __table_args__ = (
        Index("idx_discovered_rank", "rank_score"),
        Index("idx_discovered_pnl", "total_pnl"),
        Index("idx_discovered_win_rate", "win_rate"),
        Index("idx_discovered_profitable", "is_profitable"),
        Index("idx_discovered_recommendation", "recommendation"),
        Index("idx_discovered_cluster", "cluster_id"),
        Index("idx_discovered_analyzed", "last_analyzed_at"),
        Index("idx_discovered_composite", "composite_score"),
        Index("idx_discovered_in_pool", "in_top_pool"),
        Index("idx_discovered_last_trade", "last_trade_at"),
        Index("idx_discovered_insider_score", "insider_score"),
    )


class WalletTag(Base):
    """Tag definition for classifying wallets"""

    __tablename__ = "wallet_tags"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False, unique=True)  # e.g., "smart_predictor"
    display_name = Column(String, nullable=False)  # e.g., "Smart Predictor"
    description = Column(Text, nullable=True)
    category = Column(
        String, default="behavioral"
    )  # behavioral, performance, risk, strategy
    color = Column(String, default="#6B7280")  # Hex color for UI
    criteria = Column(JSON, nullable=True)  # Auto-assignment criteria
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_tag_name", "name"),
        Index("idx_tag_category", "category"),
    )


class WalletCluster(Base):
    """Group of wallets believed to belong to the same entity"""

    __tablename__ = "wallet_clusters"

    id = Column(String, primary_key=True)
    label = Column(String, nullable=True)  # Human-readable label
    confidence = Column(Float, default=0.0)  # How confident we are these are related

    # Aggregate stats across all wallets in cluster
    total_wallets = Column(Integer, default=0)
    combined_pnl = Column(Float, default=0.0)
    combined_trades = Column(Integer, default=0)
    avg_win_rate = Column(Float, default=0.0)

    # Detection method
    detection_method = Column(
        String, nullable=True
    )  # funding_source, timing_correlation, pattern_match
    evidence = Column(JSON, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (Index("idx_cluster_pnl", "combined_pnl"),)


class TraderGroup(Base):
    """User-defined or auto-suggested group of traders to monitor together."""

    __tablename__ = "trader_groups"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False, unique=True)
    description = Column(Text, nullable=True)
    source_type = Column(String, default="manual")  # manual, suggested_cluster, suggested_tag, suggested_pool
    suggestion_key = Column(String, nullable=True)
    criteria = Column(JSON, default=dict)
    auto_track_members = Column(Boolean, default=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("idx_trader_group_active", "is_active"),
        Index("idx_trader_group_source", "source_type"),
    )


class TraderGroupMember(Base):
    """Member wallet within a tracked trader group."""

    __tablename__ = "trader_group_members"

    id = Column(String, primary_key=True)
    group_id = Column(
        String, ForeignKey("trader_groups.id", ondelete="CASCADE"), nullable=False
    )
    wallet_address = Column(String, nullable=False)
    source = Column(String, default="manual")  # manual, suggested, imported
    confidence = Column(Float, nullable=True)
    notes = Column(Text, nullable=True)
    added_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("group_id", "wallet_address", name="uq_group_wallet"),
        Index("idx_trader_group_member_group", "group_id"),
        Index("idx_trader_group_member_wallet", "wallet_address"),
    )


class MarketConfluenceSignal(Base):
    """Signal generated when multiple top wallets converge on the same market"""

    __tablename__ = "market_confluence_signals"

    id = Column(String, primary_key=True)
    market_id = Column(String, nullable=False)
    market_question = Column(Text, nullable=True)
    market_slug = Column(String, nullable=True)

    # Signal details
    signal_type = Column(
        String, nullable=False
    )  # "multi_wallet_buy", "multi_wallet_sell", "accumulation"
    strength = Column(Float, default=0.0)  # 0-1 signal strength
    conviction_score = Column(Float, default=0.0)  # 0-100 signal conviction
    tier = Column(String, default="WATCH")  # WATCH, HIGH, EXTREME
    window_minutes = Column(Integer, default=60)
    wallet_count = Column(Integer, default=0)  # How many wallets are converging
    cluster_adjusted_wallet_count = Column(Integer, default=0)
    unique_core_wallets = Column(Integer, default=0)
    weighted_wallet_score = Column(Float, default=0.0)
    wallets = Column(JSON, default=list)  # List of wallet addresses involved

    # Market context
    outcome = Column(String, nullable=True)  # YES or NO
    avg_entry_price = Column(Float, nullable=True)
    total_size = Column(Float, nullable=True)  # Combined position size
    avg_wallet_rank = Column(
        Float, nullable=True
    )  # Average rank of participating wallets
    net_notional = Column(Float, nullable=True)
    conflicting_notional = Column(Float, nullable=True)
    market_liquidity = Column(Float, nullable=True)
    market_volume_24h = Column(Float, nullable=True)

    # Status
    is_active = Column(Boolean, default=True)
    first_seen_at = Column(DateTime, default=datetime.utcnow)
    last_seen_at = Column(DateTime, default=datetime.utcnow)
    detected_at = Column(DateTime, default=datetime.utcnow)
    expired_at = Column(DateTime, nullable=True)
    cooldown_until = Column(DateTime, nullable=True)

    __table_args__ = (
        Index("idx_confluence_market", "market_id"),
        Index("idx_confluence_strength", "strength"),
        Index("idx_confluence_active", "is_active"),
        Index("idx_confluence_detected", "detected_at"),
        Index("idx_confluence_tier", "tier"),
        Index("idx_confluence_last_seen", "last_seen_at"),
    )


class WalletActivityRollup(Base):
    """Event-level wallet activity used for near-real-time recency scoring and confluence windows."""

    __tablename__ = "wallet_activity_rollups"

    id = Column(String, primary_key=True)
    wallet_address = Column(String, nullable=False, index=True)
    market_id = Column(String, nullable=False, index=True)
    side = Column(String, nullable=True)  # BUY/SELL/YES/NO
    size = Column(Float, nullable=True)
    price = Column(Float, nullable=True)
    notional = Column(Float, nullable=True)
    tx_hash = Column(String, nullable=True)
    source = Column(String, default="unknown")  # ws, activity_api, trades_api, holders_api
    cluster_id = Column(String, nullable=True)
    traded_at = Column(DateTime, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_war_wallet_time", "wallet_address", "traded_at"),
        Index("idx_war_market_side_time", "market_id", "side", "traded_at"),
        Index("idx_war_source_time", "source", "traded_at"),
    )


class CrossPlatformEntity(Base):
    """Tracks a trader across multiple prediction market platforms"""

    __tablename__ = "cross_platform_entities"

    id = Column(String, primary_key=True)
    label = Column(String, nullable=True)

    # Platform identifiers
    polymarket_address = Column(String, nullable=True)
    kalshi_username = Column(String, nullable=True)

    # Cross-platform stats
    polymarket_pnl = Column(Float, default=0.0)
    kalshi_pnl = Column(Float, default=0.0)
    combined_pnl = Column(Float, default=0.0)

    # Behavioral analysis
    cross_platform_arb = Column(
        Boolean, default=False
    )  # Trades same event on both platforms
    hedging_detected = Column(Boolean, default=False)
    matching_markets = Column(JSON, default=list)  # Markets traded on both platforms

    confidence = Column(Float, default=0.0)  # Confidence that these are the same entity

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("idx_cross_platform_poly", "polymarket_address"),
        Index("idx_cross_platform_kalshi", "kalshi_username"),
        Index("idx_cross_platform_pnl", "combined_pnl"),
    )


# ==================== SHARED STATE (DB AS SINGLE SOURCE OF TRUTH) ====================


class ScannerRun(Base):
    """Immutable record of a scanner cycle."""

    __tablename__ = "scanner_runs"

    id = Column(String, primary_key=True)
    scan_mode = Column(String, nullable=False, default="full")  # full | fast | manual
    success = Column(Boolean, nullable=False, default=True)
    error = Column(Text, nullable=True)
    opportunity_count = Column(Integer, nullable=False, default=0)
    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=False)

    __table_args__ = (
        Index("idx_scanner_runs_completed", "completed_at"),
        Index("idx_scanner_runs_mode", "scan_mode"),
        Index("idx_scanner_runs_success", "success"),
    )


class OpportunityState(Base):
    """Current state for each opportunity stable_id (latest known value)."""

    __tablename__ = "opportunity_state"

    stable_id = Column(String, primary_key=True)
    opportunity_json = Column(JSON, nullable=False)
    first_seen_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_seen_at = Column(DateTime, nullable=False)
    is_active = Column(Boolean, nullable=False, default=True)
    last_run_id = Column(String, ForeignKey("scanner_runs.id"), nullable=True)

    __table_args__ = (
        Index("idx_opportunity_state_active", "is_active"),
        Index("idx_opportunity_state_last_seen", "last_seen_at"),
        Index("idx_opportunity_state_last_run", "last_run_id"),
    )


class OpportunityEvent(Base):
    """Append-only event log of opportunity lifecycle changes."""

    __tablename__ = "opportunity_events"

    id = Column(String, primary_key=True)
    stable_id = Column(String, nullable=False)
    run_id = Column(String, ForeignKey("scanner_runs.id"), nullable=False)
    event_type = Column(
        String, nullable=False
    )  # detected | updated | expired | reactivated
    opportunity_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("idx_opportunity_events_created", "created_at"),
        Index("idx_opportunity_events_stable", "stable_id"),
        Index("idx_opportunity_events_run", "run_id"),
        Index("idx_opportunity_events_type", "event_type"),
    )


class ScannerControl(Base):
    """Control flags for scanner worker (pause, request one-time scan)."""

    __tablename__ = "scanner_control"

    id = Column(String, primary_key=True, default="default")
    is_enabled = Column(Boolean, default=True)
    is_paused = Column(Boolean, default=False)
    scan_interval_seconds = Column(Integer, default=60)
    requested_scan_at = Column(DateTime, nullable=True)  # set by API to trigger one scan
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ScannerSnapshot(Base):
    """Latest scanner output: opportunities + status. Written by scanner worker, read by API."""

    __tablename__ = "scanner_snapshot"

    id = Column(String, primary_key=True, default="latest")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_scan_at = Column(DateTime, nullable=True)
    opportunities_json = Column(JSON, default=list)  # list of ArbitrageOpportunity dicts
    # Status fields (denormalized for API)
    running = Column(Boolean, default=True)
    enabled = Column(Boolean, default=True)
    current_activity = Column(String, nullable=True)
    interval_seconds = Column(Integer, default=60)
    strategies_json = Column(JSON, default=list)  # list of {name, type}
    tiered_scanning_json = Column(JSON, nullable=True)
    ws_feeds_json = Column(JSON, nullable=True)
    # market_id -> [{t: epoch_ms, yes: float, no: float}, ...]
    market_history_json = Column(JSON, default=dict)


class NewsWorkflowControl(Base):
    """Control flags for news workflow worker (pause, request one-time scan, lease)."""

    __tablename__ = "news_workflow_control"

    id = Column(String, primary_key=True, default="default")
    is_enabled = Column(Boolean, default=True)
    is_paused = Column(Boolean, default=False)
    scan_interval_seconds = Column(Integer, default=120)
    requested_scan_at = Column(DateTime, nullable=True)
    lease_owner = Column(String, nullable=True)
    lease_expires_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class NewsWorkflowSnapshot(Base):
    """Latest news workflow output/status written by worker, read by API/UI."""

    __tablename__ = "news_workflow_snapshot"

    id = Column(String, primary_key=True, default="latest")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_scan_at = Column(DateTime, nullable=True)
    next_scan_at = Column(DateTime, nullable=True)
    running = Column(Boolean, default=True)
    enabled = Column(Boolean, default=True)
    current_activity = Column(String, nullable=True)
    interval_seconds = Column(Integer, default=120)
    last_error = Column(Text, nullable=True)
    degraded_mode = Column(Boolean, default=False)
    budget_remaining_usd = Column(Float, nullable=True)
    stats_json = Column(JSON, default=dict)


class DiscoveryControl(Base):
    """Control flags for discovery worker (pause, request one-time run)."""

    __tablename__ = "discovery_control"

    id = Column(String, primary_key=True, default="default")
    is_enabled = Column(Boolean, default=True)
    is_paused = Column(Boolean, default=False)
    run_interval_minutes = Column(Integer, default=60)
    priority_backlog_mode = Column(Boolean, default=True)
    requested_run_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class DiscoverySnapshot(Base):
    """Latest discovery status. Written by discovery worker, read by API/UI."""

    __tablename__ = "discovery_snapshot"

    id = Column(String, primary_key=True, default="latest")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_run_at = Column(DateTime, nullable=True)
    running = Column(Boolean, default=False)
    enabled = Column(Boolean, default=True)
    current_activity = Column(String, nullable=True)
    run_interval_minutes = Column(Integer, default=60)
    wallets_discovered_last_run = Column(Integer, default=0)
    wallets_analyzed_last_run = Column(Integer, default=0)


class WeatherControl(Base):
    """Control flags for weather worker (pause, request one-time scan)."""

    __tablename__ = "weather_control"

    id = Column(String, primary_key=True, default="default")
    is_enabled = Column(Boolean, default=True)
    is_paused = Column(Boolean, default=False)
    scan_interval_seconds = Column(Integer, default=14400)
    requested_scan_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class WeatherSnapshot(Base):
    """Latest weather workflow output: opportunities + status."""

    __tablename__ = "weather_snapshot"

    id = Column(String, primary_key=True, default="latest")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_scan_at = Column(DateTime, nullable=True)
    opportunities_json = Column(JSON, default=list)
    running = Column(Boolean, default=True)
    enabled = Column(Boolean, default=True)
    current_activity = Column(String, nullable=True)
    interval_seconds = Column(Integer, default=14400)
    stats_json = Column(JSON, default=dict)


class WeatherTradeIntent(Base):
    """Execution-oriented weather trade intent generated from model signals."""

    __tablename__ = "weather_trade_intents"

    id = Column(String, primary_key=True)
    market_id = Column(String, nullable=False, index=True)
    market_question = Column(Text, nullable=False)
    direction = Column(String, nullable=False)  # buy_yes | buy_no
    entry_price = Column(Float, nullable=True)
    take_profit_price = Column(Float, nullable=True)
    stop_loss_pct = Column(Float, nullable=True)
    model_probability = Column(Float, nullable=True)
    edge_percent = Column(Float, nullable=True)
    confidence = Column(Float, nullable=True)
    model_agreement = Column(Float, nullable=True)
    suggested_size_usd = Column(Float, nullable=True)
    metadata_json = Column(JSON, nullable=True)
    status = Column(
        String, default="pending", nullable=False
    )  # pending | submitted | executed | skipped | expired
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    consumed_at = Column(DateTime, nullable=True)

    __table_args__ = (
        Index("idx_weather_intent_created", "created_at"),
        Index("idx_weather_intent_status", "status"),
        Index("idx_weather_intent_market", "market_id"),
    )


class InsiderTradeIntent(Base):
    """Execution-oriented insider trade intent generated from wallet behavior signals."""

    __tablename__ = "insider_trade_intents"

    id = Column(String, primary_key=True)
    market_id = Column(String, nullable=False, index=True)
    market_question = Column(Text, nullable=False)
    direction = Column(String, nullable=False)  # buy_yes | buy_no
    entry_price = Column(Float, nullable=True)
    edge_percent = Column(Float, nullable=True)
    confidence = Column(Float, nullable=True)
    insider_score = Column(Float, nullable=True)
    wallet_addresses_json = Column(JSON, default=list)
    suggested_size_usd = Column(Float, nullable=True)
    metadata_json = Column(JSON, nullable=True)
    signal_key = Column(String, nullable=True, index=True)
    status = Column(
        String, default="pending", nullable=False
    )  # pending | submitted | executed | skipped | expired
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    consumed_at = Column(DateTime, nullable=True)

    __table_args__ = (
        Index("idx_insider_intent_created", "created_at"),
        Index("idx_insider_intent_status", "status"),
        Index("idx_insider_intent_market", "market_id"),
        Index("idx_insider_intent_signal", "signal_key", unique=True),
    )


# ==================== NORMALIZED TRADE SIGNAL BUS ====================


class TradeSignal(Base):
    """Normalized cross-source trade signal emitted by domain workers."""

    __tablename__ = "trade_signals"

    id = Column(String, primary_key=True)
    source = Column(String, nullable=False, index=True)
    source_item_id = Column(String, nullable=True)
    signal_type = Column(String, nullable=False)
    strategy_type = Column(String, nullable=True)
    market_id = Column(String, nullable=False, index=True)
    market_question = Column(Text, nullable=True)
    direction = Column(String, nullable=True)  # buy_yes | buy_no | hold
    entry_price = Column(Float, nullable=True)
    effective_price = Column(Float, nullable=True)
    edge_percent = Column(Float, nullable=True)
    confidence = Column(Float, nullable=True)
    liquidity = Column(Float, nullable=True)
    expires_at = Column(DateTime, nullable=True, index=True)
    status = Column(
        String, nullable=False, default="pending"
    )  # pending | selected | submitted | executed | skipped | expired | failed
    payload_json = Column(JSON, nullable=True)
    dedupe_key = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("idx_trade_signals_created", "created_at"),
        Index("idx_trade_signals_source_status", "source", "status"),
        Index("idx_trade_signals_market_status", "market_id", "status"),
        UniqueConstraint("source", "dedupe_key", name="uq_trade_signals_source_dedupe"),
    )


class TradeSignalSnapshot(Base):
    """Aggregated signal counts/freshness per source for UI and health."""

    __tablename__ = "trade_signal_snapshots"

    source = Column(String, primary_key=True)
    pending_count = Column(Integer, default=0)
    selected_count = Column(Integer, default=0)
    submitted_count = Column(Integer, default=0)
    executed_count = Column(Integer, default=0)
    skipped_count = Column(Integer, default=0)
    expired_count = Column(Integer, default=0)
    failed_count = Column(Integer, default=0)
    latest_signal_at = Column(DateTime, nullable=True)
    oldest_pending_at = Column(DateTime, nullable=True)
    freshness_seconds = Column(Float, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    stats_json = Column(JSON, default=dict)


# ==================== WORKER RUNTIME STATE ====================


class WorkerControl(Base):
    """Generic worker control row for independently owned worker loops."""

    __tablename__ = "worker_control"

    worker_name = Column(String, primary_key=True)
    is_enabled = Column(Boolean, default=True)
    is_paused = Column(Boolean, default=False)
    interval_seconds = Column(Integer, default=60)
    requested_run_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class WorkerSnapshot(Base):
    """Latest worker status snapshot for API/websocket health surfaces."""

    __tablename__ = "worker_snapshot"

    worker_name = Column(String, primary_key=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_run_at = Column(DateTime, nullable=True)
    running = Column(Boolean, default=False)
    enabled = Column(Boolean, default=True)
    current_activity = Column(String, nullable=True)
    interval_seconds = Column(Integer, default=60)
    lag_seconds = Column(Float, nullable=True)
    last_error = Column(Text, nullable=True)
    stats_json = Column(JSON, default=dict)


# ==================== TRADER ORCHESTRATOR PERSISTENCE ====================


class TraderOrchestratorControl(Base):
    """Control flags for the dedicated trader orchestrator worker loop."""

    __tablename__ = "trader_orchestrator_control"

    id = Column(String, primary_key=True, default="default")
    is_enabled = Column(Boolean, default=False)
    is_paused = Column(Boolean, default=True)
    mode = Column(String, default="paper")
    run_interval_seconds = Column(Integer, default=2)
    requested_run_at = Column(DateTime, nullable=True)
    kill_switch = Column(Boolean, default=False)
    settings_json = Column(JSON, default=dict)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class TraderOrchestratorSnapshot(Base):
    """Latest orchestrator status/performance snapshot."""

    __tablename__ = "trader_orchestrator_snapshot"

    id = Column(String, primary_key=True, default="latest")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_run_at = Column(DateTime, nullable=True)
    running = Column(Boolean, default=False)
    enabled = Column(Boolean, default=False)
    current_activity = Column(String, nullable=True)
    interval_seconds = Column(Integer, default=2)
    traders_total = Column(Integer, default=0)
    traders_running = Column(Integer, default=0)
    decisions_count = Column(Integer, default=0)
    orders_count = Column(Integer, default=0)
    open_orders = Column(Integer, default=0)
    gross_exposure_usd = Column(Float, default=0.0)
    daily_pnl = Column(Float, default=0.0)
    last_error = Column(Text, nullable=True)
    stats_json = Column(JSON, default=dict)


class Trader(Base):
    """Single trader definition owned by the orchestrator."""

    __tablename__ = "traders"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False, unique=True, index=True)
    description = Column(Text, nullable=True)
    strategy_key = Column(String, nullable=False, index=True)
    strategy_version = Column(String, nullable=False, default="v1")
    sources_json = Column(JSON, default=list)
    params_json = Column(JSON, default=dict)
    risk_limits_json = Column(JSON, default=dict)
    metadata_json = Column(JSON, default=dict)
    is_enabled = Column(Boolean, default=True)
    is_paused = Column(Boolean, default=False)
    interval_seconds = Column(Integer, default=60)
    requested_run_at = Column(DateTime, nullable=True)
    last_run_at = Column(DateTime, nullable=True)
    next_run_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class TraderSignalCursor(Base):
    """Per-trader cursor to bound signal scans and reduce repeated range scans."""

    __tablename__ = "trader_signal_cursor"

    trader_id = Column(
        String,
        ForeignKey("traders.id", ondelete="CASCADE"),
        primary_key=True,
    )
    last_signal_created_at = Column(DateTime, nullable=True)
    last_signal_id = Column(String, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class TraderDecision(Base):
    """Decision audit log for every trader evaluation."""

    __tablename__ = "trader_decisions"

    id = Column(String, primary_key=True)
    trader_id = Column(
        String,
        ForeignKey("traders.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    signal_id = Column(
        String,
        ForeignKey("trade_signals.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    source = Column(String, nullable=False, index=True)
    strategy_key = Column(String, nullable=False, index=True)
    decision = Column(String, nullable=False)  # selected | skipped | blocked | failed
    reason = Column(Text, nullable=True)
    score = Column(Float, nullable=True)
    event_id = Column(String, nullable=True, index=True)
    trace_id = Column(String, nullable=True, index=True)
    checks_summary_json = Column(JSON, default=dict)
    risk_snapshot_json = Column(JSON, default=dict)
    payload_json = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("idx_trader_decisions_created", "created_at"),
        Index("idx_trader_decisions_decision", "decision"),
        Index("idx_trader_decisions_trader_signal", "trader_id", "signal_id"),
    )


class TraderDecisionCheck(Base):
    """Per-rule decision evaluation records for explainability."""

    __tablename__ = "trader_decision_checks"

    id = Column(String, primary_key=True)
    decision_id = Column(
        String,
        ForeignKey("trader_decisions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    check_key = Column(String, nullable=False, index=True)
    check_label = Column(String, nullable=False)
    passed = Column(Boolean, nullable=False, default=False)
    score = Column(Float, nullable=True)
    detail = Column(Text, nullable=True)
    payload_json = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("idx_trader_decision_checks_decision_created", "decision_id", "created_at"),
    )


class TraderOrder(Base):
    """Execution records owned by a trader and tied to a decision/signal."""

    __tablename__ = "trader_orders"

    id = Column(String, primary_key=True)
    trader_id = Column(
        String,
        ForeignKey("traders.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    signal_id = Column(String, ForeignKey("trade_signals.id"), nullable=True, index=True)
    decision_id = Column(
        String,
        ForeignKey("trader_decisions.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    source = Column(String, nullable=False, index=True)
    market_id = Column(String, nullable=False, index=True)
    market_question = Column(Text, nullable=True)
    direction = Column(String, nullable=True)
    event_id = Column(String, nullable=True, index=True)
    trace_id = Column(String, nullable=True, index=True)
    mode = Column(String, nullable=False, default="paper")
    status = Column(String, nullable=False, default="submitted")
    notional_usd = Column(Float, nullable=True)
    entry_price = Column(Float, nullable=True)
    effective_price = Column(Float, nullable=True)
    edge_percent = Column(Float, nullable=True)
    confidence = Column(Float, nullable=True)
    actual_profit = Column(Float, nullable=True)
    reason = Column(Text, nullable=True)
    payload_json = Column(JSON, default=dict)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    executed_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("idx_trader_orders_created", "created_at"),
        Index("idx_trader_orders_status", "status"),
        Index("idx_trader_orders_trader_created", "trader_id", "created_at"),
    )


class TraderSignalConsumption(Base):
    """Per-trader signal consumption ledger."""

    __tablename__ = "trader_signal_consumption"

    id = Column(String, primary_key=True)
    trader_id = Column(
        String,
        ForeignKey("traders.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    signal_id = Column(
        String,
        ForeignKey("trade_signals.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    decision_id = Column(
        String,
        ForeignKey("trader_decisions.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    outcome = Column(String, nullable=True)
    reason = Column(Text, nullable=True)
    payload_json = Column(JSON, default=dict)
    consumed_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        UniqueConstraint("trader_id", "signal_id", name="uq_trader_signal_consumption"),
        Index("idx_trader_signal_consumption_consumed", "consumed_at"),
    )


class TraderEvent(Base):
    """Immutable audit/event log for orchestrator and trader lifecycle events."""

    __tablename__ = "trader_events"

    id = Column(String, primary_key=True)
    trader_id = Column(
        String,
        ForeignKey("traders.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    event_type = Column(String, nullable=False, index=True)
    severity = Column(String, nullable=False, default="info")
    source = Column(String, nullable=True, index=True)
    operator = Column(String, nullable=True)
    message = Column(Text, nullable=True)
    trace_id = Column(String, nullable=True, index=True)
    payload_json = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    __table_args__ = (
        Index("idx_trader_events_type_created", "event_type", "created_at"),
    )


class TraderConfigRevision(Base):
    """Versioned orchestrator/trader snapshots for audit and rollback."""

    __tablename__ = "trader_config_revisions"

    id = Column(String, primary_key=True)
    trader_id = Column(
        String,
        ForeignKey("traders.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    operator = Column(String, nullable=True)
    reason = Column(Text, nullable=True)
    orchestrator_before_json = Column(JSON, default=dict)
    orchestrator_after_json = Column(JSON, default=dict)
    trader_before_json = Column(JSON, default=dict)
    trader_after_json = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

# ==================== WORLD INTELLIGENCE ====================


class WorldIntelligenceSignal(Base):
    """Aggregated world intelligence signal from any source."""

    __tablename__ = "world_intelligence_signals"

    id = Column(String, primary_key=True)
    signal_type = Column(String, nullable=False)  # conflict, tension, instability, convergence, anomaly, military, infrastructure
    severity = Column(Float, nullable=False, default=0.0)  # 0-1 normalized
    country = Column(String, nullable=True)
    iso3 = Column(String, nullable=True)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    title = Column(Text, nullable=False)
    description = Column(Text, nullable=True)
    source = Column(String, nullable=True)
    detected_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=True)
    metadata_json = Column(JSON, nullable=True)
    related_market_ids = Column(JSON, nullable=True)  # list of market IDs
    market_relevance_score = Column(Float, nullable=True)

    __table_args__ = (
        Index("idx_wi_signal_type", "signal_type"),
        Index("idx_wi_severity", "severity"),
        Index("idx_wi_country", "country"),
        Index("idx_wi_detected", "detected_at"),
    )


class CountryInstabilityRecord(Base):
    """Historical country instability index snapshots."""

    __tablename__ = "country_instability_records"

    id = Column(String, primary_key=True)
    country = Column(String, nullable=False)
    iso3 = Column(String, nullable=False)
    score = Column(Float, nullable=False)  # 0-100
    components = Column(JSON, nullable=True)  # sub-score breakdown
    trend = Column(String, nullable=True)  # rising/falling/stable
    computed_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("idx_cii_country", "country"),
        Index("idx_cii_iso3", "iso3"),
        Index("idx_cii_computed", "computed_at"),
        Index("idx_cii_score", "score"),
    )


class TensionPairRecord(Base):
    """Historical country-pair tension snapshots."""

    __tablename__ = "tension_pair_records"

    id = Column(String, primary_key=True)
    country_a = Column(String, nullable=False)
    country_b = Column(String, nullable=False)
    tension_score = Column(Float, nullable=False)  # 0-100
    event_count = Column(Integer, nullable=True)
    avg_goldstein_scale = Column(Float, nullable=True)
    trend = Column(String, nullable=True)
    computed_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("idx_tension_pair", "country_a", "country_b"),
        Index("idx_tension_computed", "computed_at"),
    )


class ConflictEventRecord(Base):
    """Cached ACLED conflict event data."""

    __tablename__ = "conflict_event_records"

    id = Column(String, primary_key=True)
    event_type = Column(String, nullable=False)
    sub_event_type = Column(String, nullable=True)
    country = Column(String, nullable=False)
    iso3 = Column(String, nullable=True)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    fatalities = Column(Integer, default=0)
    event_date = Column(DateTime, nullable=True)
    source = Column(String, nullable=True)
    notes = Column(Text, nullable=True)
    severity_score = Column(Float, nullable=True)
    fetched_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("idx_conflict_country", "country"),
        Index("idx_conflict_type", "event_type"),
        Index("idx_conflict_date", "event_date"),
        Index("idx_conflict_fetched", "fetched_at"),
    )


class WorldIntelligenceSnapshot(Base):
    """Worker snapshot for world intelligence collector."""

    __tablename__ = "world_intelligence_snapshots"

    id = Column(String, primary_key=True, default="latest")
    status = Column(JSON, nullable=True)
    signals_json = Column(JSON, nullable=True)  # last batch of signals
    stats = Column(JSON, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# ==================== DATABASE SETUP ====================

# SQLite-specific: improve concurrency (WAL + busy_timeout applied in _set_sqlite_pragma)
_engine_kw: dict = {"echo": False}
if "sqlite" in settings.DATABASE_URL:
    _engine_kw["connect_args"] = {"timeout": 30}  # Wait up to 30s when DB is locked
# For Postgres, pool_size/max_overflow can be set via env or here if needed

async_engine = create_async_engine(settings.DATABASE_URL, **_engine_kw)


def _set_sqlite_pragma(dbapi_connection, connection_record):
    """Configure SQLite for better concurrent access (WAL mode, busy timeout)."""
    if "sqlite" not in settings.DATABASE_URL:
        return
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")  # Allow concurrent reads during writes
    cursor.execute("PRAGMA busy_timeout=30000")  # Wait up to 30s when locked (ms)
    cursor.close()


# Apply pragmas on each new SQLite connection
event.listens_for(async_engine.sync_engine, "connect")(_set_sqlite_pragma)

AsyncSessionLocal = sessionmaker(
    async_engine, class_=AsyncSession, expire_on_commit=False
)



def _run_alembic_upgrade(connection) -> None:
    from alembic import command
    from alembic.config import Config

    backend_root = Path(__file__).resolve().parents[1]
    alembic_cfg = Config(str(backend_root / "alembic.ini"))
    alembic_cfg.set_main_option("script_location", str(backend_root / "alembic"))
    alembic_cfg.set_main_option("sqlalchemy.url", str(connection.engine.url))
    alembic_cfg.attributes["connection"] = connection
    command.upgrade(alembic_cfg, "head")


@contextmanager
def _sqlite_migration_lock():
    """Serialize Alembic upgrades across processes for SQLite databases."""
    if "sqlite" not in settings.DATABASE_URL:
        yield
        return

    lock_path = Path(__file__).resolve().parents[1] / ".alembic.sqlite.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    # Delete stale lock file if it has grown unreasonably (>4 KB).
    try:
        if lock_path.exists() and lock_path.stat().st_size > 4096:
            lock_path.unlink(missing_ok=True)
    except OSError:
        pass

    # Use "a" (append) so multiple processes can open the file without
    # Windows blocking on a competing truncate ("w" causes PermissionError
    # when 9 services race to open the same file).
    lock_file = None
    try:
        lock_file = lock_path.open("a", encoding="utf-8")
    except (PermissionError, OSError):
        # Cannot even open the lock file — proceed without a lock.
        logger.warning("Cannot open migration lock file, proceeding without lock")
        yield
        return

    try:
        if os.name == "posix":
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            return

        if os.name == "nt":
            import msvcrt
            import time

            deadline = time.monotonic() + 30
            locked = False
            while True:
                try:
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                    locked = True
                    break
                except (PermissionError, OSError):
                    if time.monotonic() >= deadline:
                        logger.warning(
                            "Could not acquire migration lock after 30s, "
                            "proceeding without lock"
                        )
                        break
                    time.sleep(0.5)
            try:
                yield
            finally:
                if locked:
                    try:
                        lock_file.seek(0)
                        msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                    except (PermissionError, OSError):
                        pass
            return

        # Unknown platform: continue without an OS-level file lock.
        yield
    finally:
        try:
            lock_file.close()
        except (PermissionError, OSError):
            pass


async def init_database():
    """Initialize database and apply Alembic migrations."""
    from models.model_registry import register_all_models

    register_all_models()
    with _sqlite_migration_lock():
        async with async_engine.begin() as conn:
            await conn.run_sync(_run_alembic_upgrade)


async def get_db_session() -> AsyncSession:
    """Get database session"""
    async with AsyncSessionLocal() as session:
        yield session
