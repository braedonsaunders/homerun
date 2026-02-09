from sqlalchemy import (
    Column,
    String,
    Float,
    Integer,
    Boolean,
    DateTime,
    Text,
    JSON,
    ForeignKey,
    Enum as SQLEnum,
    Index,
    inspect as sa_inspect,
    text,
)
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime
import enum
import logging

from config import settings

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
    source_wallet = Column(String, nullable=False, index=True)
    account_id = Column(String, ForeignKey("simulation_accounts.id"), nullable=False)

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
    llm_provider = Column(String, default="none")  # none, openai, anthropic
    llm_model = Column(String, nullable=True)
    google_api_key = Column(String, nullable=True)
    xai_api_key = Column(String, nullable=True)
    deepseek_api_key = Column(String, nullable=True)

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

    # Scanner Settings
    scan_interval_seconds = Column(Integer, default=60)
    min_profit_threshold = Column(Float, default=2.5)
    max_markets_to_scan = Column(Integer, default=500)
    min_liquidity = Column(Float, default=1000.0)

    # Trading Safety Settings
    trading_enabled = Column(Boolean, default=False)
    max_trade_size_usd = Column(Float, default=100.0)
    max_daily_trade_volume = Column(Float, default=1000.0)
    max_open_positions = Column(Integer, default=10)
    max_slippage_percent = Column(Float, default=2.0)

    # Database Maintenance
    auto_cleanup_enabled = Column(Boolean, default=False)
    cleanup_interval_hours = Column(Integer, default=24)
    cleanup_resolved_trade_days = Column(Integer, default=30)

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

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


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
    )  # openai, anthropic, google, xai, deepseek
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
    reasoning = Column(Text, nullable=True)  # Full chain-of-thought
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
    )  # openai, anthropic, google, xai, deepseek
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

    # Tags (many-to-many via JSON for simplicity in SQLite)
    tags = Column(JSON, default=list)  # ["smart_predictor", "whale", "consistent", ...]

    # Entity clustering
    cluster_id = Column(String, nullable=True)  # Which cluster this wallet belongs to

    __table_args__ = (
        Index("idx_discovered_rank", "rank_score"),
        Index("idx_discovered_pnl", "total_pnl"),
        Index("idx_discovered_win_rate", "win_rate"),
        Index("idx_discovered_profitable", "is_profitable"),
        Index("idx_discovered_recommendation", "recommendation"),
        Index("idx_discovered_cluster", "cluster_id"),
        Index("idx_discovered_analyzed", "last_analyzed_at"),
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


class MarketConfluenceSignal(Base):
    """Signal generated when multiple top wallets converge on the same market"""

    __tablename__ = "market_confluence_signals"

    id = Column(String, primary_key=True)
    market_id = Column(String, nullable=False)
    market_question = Column(Text, nullable=True)

    # Signal details
    signal_type = Column(
        String, nullable=False
    )  # "multi_wallet_buy", "multi_wallet_sell", "accumulation"
    strength = Column(Float, default=0.0)  # 0-1 signal strength
    wallet_count = Column(Integer, default=0)  # How many wallets are converging
    wallets = Column(JSON, default=list)  # List of wallet addresses involved

    # Market context
    outcome = Column(String, nullable=True)  # YES or NO
    avg_entry_price = Column(Float, nullable=True)
    total_size = Column(Float, nullable=True)  # Combined position size
    avg_wallet_rank = Column(
        Float, nullable=True
    )  # Average rank of participating wallets

    # Status
    is_active = Column(Boolean, default=True)
    detected_at = Column(DateTime, default=datetime.utcnow)
    expired_at = Column(DateTime, nullable=True)

    __table_args__ = (
        Index("idx_confluence_market", "market_id"),
        Index("idx_confluence_strength", "strength"),
        Index("idx_confluence_active", "is_active"),
        Index("idx_confluence_detected", "detected_at"),
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


# ==================== DATABASE SETUP ====================

async_engine = create_async_engine(settings.DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(
    async_engine, class_=AsyncSession, expire_on_commit=False
)


def _get_column_default_sql(col):
    """Get SQL default value for a column."""
    if col.default is not None:
        val = col.default.arg
        if callable(val):
            return None
        if isinstance(val, bool):
            return "1" if val else "0"
        if isinstance(val, (int, float)):
            return str(val)
        if isinstance(val, str):
            return f"'{val}'"
        if isinstance(val, enum.Enum):
            return f"'{val.name}'"
    return None


def _get_column_type_sql(col):
    """Get SQL type string for a column."""
    type_obj = col.type
    type_name = type_obj.__class__.__name__.upper()
    type_map = {
        "STRING": "VARCHAR",
        "TEXT": "TEXT",
        "INTEGER": "INTEGER",
        "FLOAT": "FLOAT",
        "BOOLEAN": "BOOLEAN",
        "DATETIME": "DATETIME",
        "JSON": "JSON",
        "ENUM": "VARCHAR",
    }
    return type_map.get(type_name, "VARCHAR")


def _fix_enum_values(connection):
    """Fix enum columns that were stored with .value instead of .name."""
    # Map of (table, column) -> {wrong_value: correct_name}
    enum_fixes = {
        ("copy_trading_configs", "copy_mode"): {
            "all_trades": "ALL_TRADES",
            "arb_only": "ARB_ONLY",
        },
    }
    inspector = sa_inspect(connection)
    existing_tables = set(inspector.get_table_names())

    for (table_name, col_name), value_map in enum_fixes.items():
        if table_name not in existing_tables:
            continue
        existing_cols = {c["name"] for c in inspector.get_columns(table_name)}
        if col_name not in existing_cols:
            continue
        for wrong_val, correct_val in value_map.items():
            connection.execute(
                text(
                    f"UPDATE {table_name} SET {col_name} = :correct WHERE {col_name} = :wrong"
                ),
                {"correct": correct_val, "wrong": wrong_val},
            )


def _migrate_schema(connection):
    """Add missing columns to existing tables (SQLite ALTER TABLE ADD COLUMN)."""
    inspector = sa_inspect(connection)
    existing_tables = inspector.get_table_names()

    for table in Base.metadata.sorted_tables:
        if table.name not in existing_tables:
            continue

        existing_cols = {c["name"] for c in inspector.get_columns(table.name)}
        for col in table.columns:
            if col.name in existing_cols:
                continue

            col_type = _get_column_type_sql(col)
            default_sql = _get_column_default_sql(col)

            stmt = f"ALTER TABLE {table.name} ADD COLUMN {col.name} {col_type}"
            if default_sql is not None:
                stmt += f" DEFAULT {default_sql}"

            logger.info(f"Migrating: {stmt}")
            connection.execute(text(stmt))

    _fix_enum_values(connection)


async def init_database():
    """Initialize database tables and migrate schema for existing databases."""
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(_migrate_schema)


async def get_db_session() -> AsyncSession:
    """Get database session"""
    async with AsyncSessionLocal() as session:
        yield session
