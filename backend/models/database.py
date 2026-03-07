from sqlalchemy import (
    Column,
    String,
    Integer,
    Boolean,
    Text,
    JSON,
    ForeignKey,
    Enum as SQLEnum,
    Index,
    UniqueConstraint,
    event as _sa_event,
)
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.exc import DBAPIError
from sqlalchemy.types import TypeDecorator, DateTime as SADateTime
from datetime import datetime, timezone
from pathlib import Path
import logging as _logging
import enum
import asyncio
import os as _os
import time as _time

from config import settings
from models.types import PreciseFloat as Float

Base = declarative_base()


class UTCDateTime(TypeDecorator):
    impl = SADateTime
    cache_ok = True

    def load_dialect_impl(self, dialect):
        return dialect.type_descriptor(SADateTime(timezone=False))

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if not isinstance(value, datetime):
            raise TypeError(f"UTCDateTime only accepts datetime values, got {type(value)!r}")
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).replace(tzinfo=None)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)


DateTime = UTCDateTime


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class RetryableAsyncSession(AsyncSession):
    _COMMIT_RETRY_ATTEMPTS = 4
    _COMMIT_BASE_DELAY_SECONDS = 0.05
    _COMMIT_RETRYABLE_MESSAGES = (
        "deadlock detected",
        "serialization failure",
        "could not serialize access",
        "lock not available",
        "too many clients already",
        "remaining connection slots are reserved",
        "cannot connect now",
        "connection is closed",
        "underlying connection is closed",
        "connection has been closed",
        "closed the connection unexpectedly",
        "terminating connection",
        "connection reset by peer",
        "broken pipe",
    )

    # ------------------------------------------------------------------
    # Background cleanup tasks – prevent GC from collecting the tasks
    # before they complete (the event loop only weakly references tasks).
    # ------------------------------------------------------------------
    _cleanup_tasks: set = set()

    @classmethod
    def _fire_and_forget(cls, coro) -> None:
        """Schedule *coro* as a background task that cannot be cancelled."""
        task = asyncio.create_task(coro)
        cls._cleanup_tasks.add(task)
        task.add_done_callback(cls._cleanup_tasks.discard)

    # ------------------------------------------------------------------
    # Session lifecycle – close / rollback / invalidate
    # ------------------------------------------------------------------

    async def rollback(self) -> None:
        try:
            await super().rollback()
        except asyncio.CancelledError:
            # Ensure rollback completes even if we are cancelled.
            self._fire_and_forget(self._do_rollback_or_invalidate())
            raise
        except Exception:
            try:
                await super().invalidate()
            except Exception:
                pass
            raise

    async def close(self) -> None:
        """Return the underlying connection to the pool.  Never raises.

        If the calling task is cancelled mid-close, the cleanup task
        continues in the background so the connection is *always*
        returned to the pool (or invalidated).
        """
        # Create the cleanup task and hold a strong reference so it
        # survives GC even if the calling task is cancelled.
        task = asyncio.create_task(self._do_close_or_invalidate())
        self._cleanup_tasks.add(task)
        task.add_done_callback(self._cleanup_tasks.discard)
        try:
            await asyncio.shield(task)
        except (asyncio.CancelledError, Exception):
            # task keeps running — connection WILL be returned.
            pass

    async def invalidate(self) -> None:
        try:
            await super().invalidate()
        except asyncio.CancelledError:
            self._fire_and_forget(self._do_invalidate())
            raise
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internal helpers (run inside fire-and-forget tasks)
    # ------------------------------------------------------------------

    async def _do_close_or_invalidate(self) -> None:
        """Close the session; fall back to invalidate on any error."""
        try:
            await super().close()
        except Exception:
            try:
                await super().invalidate()
            except Exception:
                pass

    async def _do_rollback_or_invalidate(self) -> None:
        """Rollback; fall back to invalidate on any error."""
        try:
            await super().rollback()
        except Exception:
            try:
                await super().invalidate()
            except Exception:
                pass

    async def _do_invalidate(self) -> None:
        """Invalidate, swallowing errors."""
        try:
            await super().invalidate()
        except Exception:
            pass

    def __del__(self) -> None:
        """Last-resort safety net: if GC is collecting this session without
        a prior close(), schedule an async cleanup on the running loop so
        the underlying connection is returned to the pool (or invalidated)
        instead of being silently leaked.
        """
        # sync_session is None once properly closed — skip if already clean.
        sync = getattr(self, "sync_session", None)
        if sync is None:
            return
        # Only act if the sync session still holds a connection.
        # SQLAlchemy 2.x stores the active connection in _connection.
        if not getattr(sync, "_connection", None):
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No event loop — force-close the sync session directly.
            try:
                sync.close()
            except Exception:
                pass
            return
        if loop.is_closed():
            try:
                sync.close()
            except Exception:
                pass
            return
        # Schedule async cleanup on the running loop.  _fire_and_forget
        # stores a strong reference so the task survives GC.
        try:
            loop.call_soon_threadsafe(self._fire_and_forget, self._do_close_or_invalidate())
        except RuntimeError:
            # Loop may have been closed between the check and the call.
            try:
                sync.close()
            except Exception:
                pass

    async def _reset_after_failed_commit(self) -> None:
        try:
            await self.rollback()
        except Exception:
            try:
                await self.invalidate()
            except Exception:
                pass

    async def commit(self) -> None:
        for attempt in range(1, self._COMMIT_RETRY_ATTEMPTS + 1):
            try:
                await super().commit()
                return
            except DBAPIError as exc:
                message = str(getattr(exc, "orig", exc)).lower()
                retryable = any(marker in message for marker in self._COMMIT_RETRYABLE_MESSAGES)
                if not retryable or attempt >= self._COMMIT_RETRY_ATTEMPTS:
                    raise
                await self._reset_after_failed_commit()
                delay = min(self._COMMIT_BASE_DELAY_SECONDS * (2 ** (attempt - 1)), 0.4)
                await asyncio.sleep(delay)

class TradeStatus(enum.Enum):
    PENDING = "pending"
    OPEN = "open"
    CLOSED_WIN = "closed_win"
    CLOSED_LOSS = "closed_loss"
    RESOLVED_WIN = "resolved_win"
    RESOLVED_LOSS = "resolved_loss"
    CANCELLED = "cancelled"
    FAILED = "failed"


class PositionSide(enum.Enum):
    YES = "yes"
    NO = "no"


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
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)

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
    opened_at = Column(DateTime, default=_utcnow)
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
    executed_at = Column(DateTime, default=_utcnow)
    resolved_at = Column(DateTime, nullable=True)

    # Copy trading reference
    copied_from_wallet = Column(String, nullable=True)

    account = relationship("SimulationAccount", back_populates="trades")

    __table_args__ = (
        Index("idx_trade_account", "account_id"),
        Index("idx_trade_status", "status"),
        Index("idx_trade_copied", "copied_from_wallet"),
    )


# ==================== WALLET ANALYSIS ====================


class TrackedWallet(Base):
    """Wallet being tracked for analysis"""

    __tablename__ = "tracked_wallets"

    address = Column(String, primary_key=True)
    label = Column(String)
    added_at = Column(DateTime, default=_utcnow)

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
    wallet_address = Column(String, ForeignKey("tracked_wallets.address"), nullable=False)

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
    detected_at = Column(DateTime, default=_utcnow)
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
    fetched_at = Column(DateTime, default=_utcnow, nullable=False)
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
    last_seen_at = Column(DateTime, default=_utcnow, nullable=False)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)

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
    created_at = Column(DateTime, default=_utcnow, nullable=False)

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
    status = Column(String, default="pending", nullable=False)  # pending | submitted | executed | skipped | expired
    created_at = Column(DateTime, default=_utcnow, nullable=False)
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
    detected_at = Column(DateTime, default=_utcnow)

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
    weights = Column(JSON, nullable=False)  # Model parameters (weights, bias, thresholds)
    feature_names = Column(JSON, nullable=False)  # Ordered list of feature names
    metrics = Column(JSON, nullable=True)  # accuracy, precision, recall, f1
    training_samples = Column(Integer, default=0)
    created_at = Column(DateTime, default=_utcnow)
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
    predicted_at = Column(DateTime, default=_utcnow)

    # Outcome tracking (filled in later)
    actual_outcome = Column(Boolean, nullable=True)
    actual_roi = Column(Float, nullable=True)

    __table_args__ = (
        Index("idx_ml_pred_opp", "opportunity_id"),
        Index("idx_ml_pred_time", "predicted_at"),
    )


# ==================== ML TRAINING DATA ====================


class MLTrainingSnapshot(Base):
    """Time-series orderbook snapshots recorded from live crypto markets for ML training.

    Captured by the crypto worker every N seconds, storing price, spread, depth,
    and volume features that can be used to train directional prediction models.
    """

    __tablename__ = "ml_training_snapshots"

    id = Column(String, primary_key=True)
    asset = Column(String(8), nullable=False)  # btc, eth, sol, xrp
    timeframe = Column(String(8), nullable=False)  # 5m, 15m, 1h, 4h
    timestamp = Column(DateTime, nullable=False)

    # Prices
    mid_price = Column(Float, nullable=False)  # (up_price + (1 - down_price)) / 2
    up_price = Column(Float, nullable=True)
    down_price = Column(Float, nullable=True)
    best_bid = Column(Float, nullable=True)
    best_ask = Column(Float, nullable=True)
    spread = Column(Float, nullable=True)  # ask - bid in cents
    combined = Column(Float, nullable=True)  # up + down (arb indicator)

    # Depth & liquidity
    liquidity = Column(Float, nullable=True)  # total market liquidity USD
    volume = Column(Float, nullable=True)  # cumulative volume
    volume_24h = Column(Float, nullable=True)  # rolling 24h volume

    # Oracle
    oracle_price = Column(Float, nullable=True)  # Chainlink BTC/ETH/SOL/XRP price
    price_to_beat = Column(Float, nullable=True)  # resolution threshold price

    # Market context
    seconds_left = Column(Integer, nullable=True)  # seconds until market resolves
    is_live = Column(Boolean, nullable=True)  # is market currently active

    __table_args__ = (
        Index("idx_mlt_asset_tf_ts", "asset", "timeframe", "timestamp"),
        Index("idx_mlt_timestamp", "timestamp"),
        Index("idx_mlt_asset", "asset"),
    )


class MLTrainedModel(Base):
    """Trained ML model artifacts for crypto directional prediction.

    Stores XGBoost/LightGBM model weights, feature definitions, metrics,
    and promotion status. Strategies load the active model at runtime.
    """

    __tablename__ = "ml_trained_models"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)  # e.g. "crypto_directional_v3"
    model_type = Column(String, nullable=False)  # xgboost, lightgbm, logistic
    version = Column(Integer, nullable=False, default=1)
    status = Column(String, nullable=False, default="trained")  # trained, active, archived

    # Model data
    weights_json = Column(JSON, nullable=False)  # serialized model weights/trees
    feature_names = Column(JSON, nullable=False)  # ordered feature list
    hyperparams = Column(JSON, nullable=True)  # training hyperparameters

    # Scope
    assets = Column(JSON, nullable=False)  # ["btc", "eth", "sol", "xrp"]
    timeframes = Column(JSON, nullable=False)  # ["5m", "15m", "1h", "4h"]

    # Training metrics
    train_accuracy = Column(Float, nullable=True)
    test_accuracy = Column(Float, nullable=True)
    test_auc = Column(Float, nullable=True)
    feature_importance = Column(JSON, nullable=True)  # {feature: importance}
    train_samples = Column(Integer, default=0)
    test_samples = Column(Integer, default=0)
    training_date_range = Column(JSON, nullable=True)  # {"start": iso, "end": iso}

    # Walk-forward validation
    walkforward_results = Column(JSON, nullable=True)  # [{fold, train_acc, test_acc, auc}]

    # Metadata
    created_at = Column(DateTime, default=_utcnow)
    promoted_at = Column(DateTime, nullable=True)
    notes = Column(Text, nullable=True)

    __table_args__ = (
        Index("idx_mlm_status", "status"),
        Index("idx_mlm_created", "created_at"),
        UniqueConstraint("name", "version", name="uq_ml_model_name_version"),
    )


class MLRecorderConfig(Base):
    """Persistent configuration for the ML data recorder.

    Stores whether recording is active, the recording interval,
    retention policy, and schedule settings.
    """

    __tablename__ = "ml_recorder_config"

    id = Column(String, primary_key=True, default="default")
    is_recording = Column(Boolean, nullable=False, default=False)
    interval_seconds = Column(Integer, nullable=False, default=60)  # how often to snapshot
    retention_days = Column(Integer, nullable=False, default=90)  # auto-prune older than this
    assets = Column(JSON, nullable=False, default=lambda: ["btc", "eth", "sol", "xrp"])
    timeframes = Column(JSON, nullable=False, default=lambda: ["5m", "15m", "1h", "4h"])

    # Schedule (null = always record when enabled)
    schedule_enabled = Column(Boolean, nullable=False, default=False)
    schedule_start_utc = Column(String, nullable=True)  # "08:00"
    schedule_end_utc = Column(String, nullable=True)  # "22:00"

    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)


class MLTrainingJob(Base):
    """Tracks ML training job executions (manual or scheduled)."""

    __tablename__ = "ml_training_jobs"

    id = Column(String, primary_key=True)
    status = Column(String, nullable=False, default="queued")  # queued, running, completed, failed
    model_type = Column(String, nullable=False, default="xgboost")  # xgboost, lightgbm
    assets = Column(JSON, nullable=False)
    timeframes = Column(JSON, nullable=False)

    # Progress
    progress = Column(Float, default=0.0)
    message = Column(String, nullable=True)
    error = Column(Text, nullable=True)

    # Results (filled on completion)
    trained_model_id = Column(String, nullable=True)  # FK to ml_trained_models.id
    result_summary = Column(JSON, nullable=True)

    # Timing
    created_at = Column(DateTime, default=_utcnow, nullable=False)
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)

    __table_args__ = (
        Index("idx_mljob_status", "status"),
        Index("idx_mljob_created", "created_at"),
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
    created_at = Column(DateTime, default=_utcnow)


class ValidationJob(Base):
    """Persistent async validation job queue (backtests/optimization)."""

    __tablename__ = "validation_jobs"

    id = Column(String, primary_key=True)
    job_type = Column(String, nullable=False)  # backtest | optimize | execution_simulation
    status = Column(String, nullable=False, default="queued")  # queued | running | completed | failed | cancelled
    payload = Column(JSON, nullable=True)
    result = Column(JSON, nullable=True)
    error = Column(Text, nullable=True)
    progress = Column(Float, default=0.0)
    message = Column(String, nullable=True)
    created_at = Column(DateTime, default=_utcnow, nullable=False)
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
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)

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
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)


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
    llm_provider = Column(String, default="none")  # none, openai, anthropic, google, xai, deepseek, ollama, lmstudio
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
    ai_resolution_analysis = Column(Boolean, default=True)  # Auto-analyze resolution criteria
    ai_opportunity_scoring = Column(Boolean, default=True)  # LLM-as-judge scoring
    ai_news_sentiment = Column(Boolean, default=True)  # News/sentiment analysis
    ai_max_monthly_spend = Column(Float, default=50.0)  # Monthly LLM cost cap
    ai_default_model = Column(String, default="gpt-4o-mini")  # Default model for AI tasks
    ai_premium_model = Column(String, default="gpt-4o")  # Model for high-value analysis

    # Notification Settings
    telegram_bot_token = Column(String, nullable=True)
    telegram_chat_id = Column(String, nullable=True)
    notifications_enabled = Column(Boolean, default=False)
    notify_on_opportunity = Column(Boolean, default=True)
    notify_on_trade = Column(Boolean, default=True)
    notify_min_roi = Column(Float, default=5.0)
    notify_autotrader_orders = Column(Boolean, default=False)
    notify_autotrader_closes = Column(Boolean, default=True)
    notify_autotrader_issues = Column(Boolean, default=True)
    notify_autotrader_timeline = Column(Boolean, default=True)
    notify_autotrader_summary_interval_minutes = Column(Integer, default=60)
    notify_autotrader_summary_per_trader = Column(Boolean, default=False)

    # Scanner Settings
    scan_interval_seconds = Column(Integer, default=60)
    min_profit_threshold = Column(Float, default=2.5)
    max_markets_to_scan = Column(Integer, default=0)
    max_events_to_scan = Column(Integer, default=0)
    market_fetch_page_size = Column(Integer, default=200)
    market_fetch_order = Column(String, default="volume")
    min_liquidity = Column(Float, default=1000.0)
    scanner_max_opportunities_total = Column(Integer, default=500)
    scanner_max_opportunities_per_strategy = Column(Integer, default=120)

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
    discovery_pool_recompute_mode = Column(String, default="quality_only")
    discovery_pool_target_size = Column(Integer, default=500)
    discovery_pool_min_size = Column(Integer, default=400)
    discovery_pool_max_size = Column(Integer, default=600)
    discovery_pool_active_window_hours = Column(Integer, default=72)
    discovery_pool_inactive_rising_retention_hours = Column(Integer, default=336)
    discovery_pool_selection_score_floor = Column(Float, default=0.55)
    discovery_pool_max_hourly_replacement_rate = Column(Float, default=0.15)
    discovery_pool_replacement_score_cutoff = Column(Float, default=0.05)
    discovery_pool_max_cluster_share = Column(Float, default=0.08)
    discovery_pool_high_conviction_threshold = Column(Float, default=0.72)
    discovery_pool_insider_priority_threshold = Column(Float, default=0.62)
    discovery_pool_min_eligible_trades = Column(Integer, default=50)
    discovery_pool_max_eligible_anomaly = Column(Float, default=0.5)
    discovery_pool_core_min_win_rate = Column(Float, default=0.60)
    discovery_pool_core_min_sharpe = Column(Float, default=1.0)
    discovery_pool_core_min_profit_factor = Column(Float, default=1.5)
    discovery_pool_rising_min_win_rate = Column(Float, default=0.55)
    discovery_pool_slo_min_analyzed_pct = Column(Float, default=95.0)
    discovery_pool_slo_min_profitable_pct = Column(Float, default=80.0)
    discovery_pool_leaderboard_wallet_trade_sample = Column(Integer, default=160)
    discovery_pool_incremental_wallet_trade_sample = Column(Integer, default=80)
    discovery_pool_full_sweep_interval_seconds = Column(Integer, default=1800)
    discovery_pool_incremental_refresh_interval_seconds = Column(Integer, default=120)
    discovery_pool_activity_reconciliation_interval_seconds = Column(Integer, default=120)
    discovery_pool_recompute_interval_seconds = Column(Integer, default=60)

    # Trading Safety Settings
    max_trade_size_usd = Column(Float, default=100.0)
    max_daily_trade_volume = Column(Float, default=1000.0)
    max_open_positions = Column(Integer, default=10)
    max_slippage_percent = Column(Float, default=2.0)
    min_account_balance_usd = Column(Float, default=0.0)

    # Opportunity Search Filters (hard rejection thresholds)
    min_liquidity_hard = Column(Float, default=1000.0)
    min_position_size = Column(Float, default=50.0)
    min_absolute_profit = Column(Float, default=10.0)
    min_annualized_roi = Column(Float, default=10.0)
    max_resolution_months = Column(Integer, default=18)
    max_plausible_roi = Column(Float, default=30.0)
    max_trade_legs = Column(Integer, default=6)
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
    btc_eth_hf_series_btc_5m = Column(String, default="10684")
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
    cleanup_trade_signal_emission_days = Column(Integer, default=21)
    cleanup_trade_signal_update_days = Column(Integer, default=3)
    cleanup_wallet_activity_rollup_days = Column(Integer, default=60)
    cleanup_wallet_activity_dedupe_enabled = Column(Boolean, default=True)
    llm_usage_retention_days = Column(Integer, default=30)
    market_cache_hygiene_enabled = Column(Boolean, default=True)
    market_cache_hygiene_interval_hours = Column(Integer, default=6)
    market_cache_retention_days = Column(Integer, default=120)
    market_cache_reference_lookback_days = Column(Integer, default=45)
    market_cache_weak_entry_grace_days = Column(Integer, default=7)
    market_cache_max_entries_per_slug = Column(Integer, default=3)

    # Trading VPN/Proxy (routes ONLY trading requests through proxy)
    trading_proxy_enabled = Column(Boolean, default=False)
    trading_proxy_url = Column(String, nullable=True)  # socks5://host:port, http://host:port
    trading_proxy_verify_ssl = Column(Boolean, default=True)
    trading_proxy_timeout = Column(Float, default=30.0)
    trading_proxy_require_vpn = Column(Boolean, default=True)  # Block trades if VPN unreachable

    # Local UI lock settings
    ui_lock_enabled = Column(Boolean, default=False)
    ui_lock_password_hash = Column(String, nullable=True)
    ui_lock_idle_timeout_minutes = Column(Integer, default=15)

    # Network access (allow LAN devices to reach the dashboard)
    allow_network_access = Column(Boolean, default=False)

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
    events_settings_json = Column(JSON, default=dict)
    events_acled_api_key = Column(String, nullable=True)
    events_acled_email = Column(String, nullable=True)
    events_opensky_username = Column(String, nullable=True)
    events_opensky_password = Column(String, nullable=True)
    events_aisstream_api_key = Column(String, nullable=True)
    events_cloudflare_radar_token = Column(String, nullable=True)
    events_country_reference_json = Column(JSON, default=list)
    events_country_reference_source = Column(String, nullable=True)
    events_country_reference_synced_at = Column(DateTime, nullable=True)
    events_ucdp_active_wars_json = Column(JSON, default=list)
    events_ucdp_minor_conflicts_json = Column(JSON, default=list)
    events_ucdp_source = Column(String, nullable=True)
    events_ucdp_year = Column(Integer, nullable=True)
    events_ucdp_synced_at = Column(DateTime, nullable=True)
    events_mid_iso3_json = Column(JSON, default=dict)
    events_mid_source = Column(String, nullable=True)
    events_mid_synced_at = Column(DateTime, nullable=True)
    events_trade_dependencies_json = Column(JSON, default=dict)
    events_trade_dependency_source = Column(String, nullable=True)
    events_trade_dependency_year = Column(Integer, nullable=True)
    events_trade_dependency_synced_at = Column(DateTime, nullable=True)
    events_chokepoints_json = Column(JSON, default=list)
    events_chokepoints_source = Column(String, nullable=True)
    events_chokepoints_synced_at = Column(DateTime, nullable=True)
    events_gdelt_news_enabled = Column(Boolean, default=True)
    events_gdelt_news_queries_json = Column(JSON, default=list)
    events_gdelt_news_timespan_hours = Column(Integer, default=6)
    events_gdelt_news_max_records = Column(Integer, default=40)
    events_gdelt_news_source = Column(String, nullable=True)
    events_gdelt_news_synced_at = Column(DateTime, nullable=True)

    # Independent Weather Workflow (forecast consensus -> opportunities/intents)
    weather_workflow_enabled = Column(Boolean, default=True)
    weather_workflow_auto_run = Column(Boolean, default=True)
    weather_workflow_scan_interval_seconds = Column(Integer, default=14400)
    weather_workflow_entry_max_price = Column(Float, default=0.92)
    weather_workflow_take_profit_price = Column(Float, default=0.85)
    weather_workflow_stop_loss_pct = Column(Float, default=50.0)
    weather_workflow_min_edge_percent = Column(Float, default=2.0)
    weather_workflow_min_confidence = Column(Float, default=0.3)
    weather_workflow_min_model_agreement = Column(Float, default=0.75)
    weather_workflow_min_liquidity = Column(Float, default=500.0)
    weather_workflow_max_markets_per_scan = Column(Integer, default=200)
    weather_workflow_default_size_usd = Column(Float, default=10.0)
    weather_workflow_max_size_usd = Column(Float, default=50.0)
    weather_workflow_model = Column(String, nullable=True)
    weather_workflow_temperature_unit = Column(String, default="F")

    # Timestamps
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)


# ==================== STRATEGY PLUGINS ====================


class StrategyTombstone(Base):
    """Permanent suppression records for seeded system strategies.

    If a system strategy slug is tombstoned, seed routines will not recreate it.
    """

    __tablename__ = "strategy_tombstones"

    slug = Column(String, primary_key=True)  # Tombstoned system strategy slug
    deleted_at = Column(DateTime, default=_utcnow, nullable=False)
    reason = Column(String, nullable=True)

    __table_args__ = (Index("idx_strategy_tombstones_deleted_at", "deleted_at"),)


class Strategy(Base):
    """Unified strategy definition — one class handles detect → evaluate → exit.

    Replaces both StrategyPlugin (detection) and TraderStrategyDefinition (execution).
    Each row is a complete Python strategy with optional detect(), evaluate(), and
    should_exit() methods.
    """

    __tablename__ = "strategies"

    id = Column(String, primary_key=True)  # UUID
    slug = Column(String, unique=True, nullable=False)  # Unique identifier
    source_key = Column(String, nullable=False, default="scanner")  # scanner, news, crypto, weather, traders
    name = Column(String, nullable=False)  # Display name
    description = Column(Text, nullable=True)
    source_code = Column(Text, nullable=False)  # Full Python source
    class_name = Column(String, nullable=True)  # Strategy class name
    is_system = Column(Boolean, default=False, nullable=False)  # Seeded built-in
    enabled = Column(Boolean, default=True)
    status = Column(String, default="unloaded")  # unloaded, loaded, error
    error_message = Column(Text, nullable=True)
    config = Column(JSON, default=dict)  # Merged config (detect + execute + exit params)
    config_schema = Column(JSON, default=dict)  # Param schema for UI form
    aliases = Column(JSON, default=list)  # Alternative slug names
    version = Column(Integer, default=1)
    sort_order = Column(Integer, default=0)
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)

    __table_args__ = (
        Index("idx_strategy_slug", "slug"),
        Index("idx_strategy_source_key", "source_key"),
        Index("idx_strategy_enabled", "enabled"),
        Index("idx_strategy_is_system", "is_system"),
        Index("idx_strategy_status", "status"),
    )


class StrategyVersion(Base):
    """Immutable strategy snapshots for versioned rollbacks and pinning."""

    __tablename__ = "strategy_versions"

    id = Column(String, primary_key=True)
    strategy_id = Column(
        String,
        ForeignKey("strategies.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    strategy_slug = Column(String, nullable=False, index=True)
    source_key = Column(String, nullable=False, default="scanner")
    version = Column(Integer, nullable=False)
    is_latest = Column(Boolean, nullable=False, default=False)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    source_code = Column(Text, nullable=False)
    class_name = Column(String, nullable=True)
    config = Column(JSON, default=dict)
    config_schema = Column(JSON, default=dict)
    aliases = Column(JSON, default=list)
    enabled = Column(Boolean, default=True)
    is_system = Column(Boolean, default=False, nullable=False)
    sort_order = Column(Integer, default=0)
    parent_version = Column(Integer, nullable=True)
    created_by = Column(String, nullable=True)
    reason = Column(Text, nullable=True)
    created_at = Column(DateTime, default=_utcnow, nullable=False)

    __table_args__ = (
        UniqueConstraint("strategy_id", "version", name="uq_strategy_versions_strategy_version"),
        Index("idx_strategy_versions_slug_version", "strategy_slug", "version"),
        Index("idx_strategy_versions_strategy_created", "strategy_id", "created_at"),
        Index("idx_strategy_versions_latest", "strategy_id", "is_latest"),
    )


class StrategyRuntimeRevision(Base):
    """Revision counters used by workers for strategy hot-reload polling."""

    __tablename__ = "strategy_runtime_revisions"

    scope = Column(String, primary_key=True)  # "__all__" or source_key
    revision = Column(Integer, nullable=False, default=0)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)

    __table_args__ = (Index("idx_strategy_runtime_revisions_updated", "updated_at"),)


# ==================== DATA SOURCES ====================


class DataSourceTombstone(Base):
    """Permanent suppression records for seeded system data sources."""

    __tablename__ = "data_source_tombstones"

    slug = Column(String, primary_key=True)
    deleted_at = Column(DateTime, default=_utcnow, nullable=False)
    reason = Column(String, nullable=True)

    __table_args__ = (Index("idx_data_source_tombstones_deleted_at", "deleted_at"),)


class DataSource(Base):
    """Unified data-source definition for pluggable ingestion/transform pipelines."""

    __tablename__ = "data_sources"

    id = Column(String, primary_key=True)
    slug = Column(String, unique=True, nullable=False)
    source_key = Column(String, nullable=False, default="custom")
    source_kind = Column(String, nullable=False, default="python")  # python | rss | rest_api
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    source_code = Column(Text, nullable=False, default="")
    class_name = Column(String, nullable=True)
    is_system = Column(Boolean, default=False, nullable=False)
    enabled = Column(Boolean, default=True)
    status = Column(String, default="unloaded")  # unloaded | loaded | error
    error_message = Column(Text, nullable=True)
    retention = Column(JSON, default=dict, nullable=False)
    config = Column(JSON, default=dict)
    config_schema = Column(JSON, default=dict)
    version = Column(Integer, default=1)
    sort_order = Column(Integer, default=0)
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)

    __table_args__ = (
        Index("idx_data_source_slug", "slug"),
        Index("idx_data_source_source_key", "source_key"),
        Index("idx_data_source_source_kind", "source_kind"),
        Index("idx_data_source_enabled", "enabled"),
        Index("idx_data_source_is_system", "is_system"),
        Index("idx_data_source_status", "status"),
    )


class DataSourceRun(Base):
    """Execution history for source runs."""

    __tablename__ = "data_source_runs"

    id = Column(String, primary_key=True)
    data_source_id = Column(String, ForeignKey("data_sources.id", ondelete="CASCADE"), nullable=False)
    source_slug = Column(String, nullable=False)
    status = Column(String, nullable=False, default="success")  # success | error
    fetched_count = Column(Integer, nullable=False, default=0)
    transformed_count = Column(Integer, nullable=False, default=0)
    upserted_count = Column(Integer, nullable=False, default=0)
    skipped_count = Column(Integer, nullable=False, default=0)
    error_message = Column(Text, nullable=True)
    metadata_json = Column(JSON, nullable=True)
    started_at = Column(DateTime, nullable=False, default=_utcnow)
    completed_at = Column(DateTime, nullable=True)
    duration_ms = Column(Integer, nullable=True)

    __table_args__ = (
        Index("idx_data_source_runs_source_slug", "source_slug"),
        Index("idx_data_source_runs_started_at", "started_at"),
        Index("idx_data_source_runs_status", "status"),
        Index("ix_data_source_runs_data_source_id", "data_source_id"),
    )


class DataSourceRecord(Base):
    """Normalized output rows produced by data-source runs."""

    __tablename__ = "data_source_records"

    id = Column(String, primary_key=True)
    data_source_id = Column(String, ForeignKey("data_sources.id", ondelete="CASCADE"), nullable=False)
    source_slug = Column(String, nullable=False)
    external_id = Column(String, nullable=True)
    title = Column(Text, nullable=True)
    summary = Column(Text, nullable=True)
    category = Column(String, nullable=True)
    source = Column(String, nullable=True)
    url = Column(Text, nullable=True)
    geotagged = Column(Boolean, default=False, nullable=False)
    country_iso3 = Column(String, nullable=True)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    observed_at = Column(DateTime, nullable=True)
    ingested_at = Column(DateTime, nullable=False, default=_utcnow)
    payload_json = Column(JSON, nullable=True)
    transformed_json = Column(JSON, nullable=True)
    tags_json = Column(JSON, nullable=True)

    __table_args__ = (
        Index("idx_data_source_records_source_slug", "source_slug"),
        Index("idx_data_source_records_data_source_id", "data_source_id"),
        Index("idx_data_source_records_observed_at", "observed_at"),
        Index("idx_data_source_records_ingested_at", "ingested_at"),
        Index("idx_data_source_records_geotagged", "geotagged"),
        Index("idx_data_source_records_country", "country_iso3"),
        Index("idx_data_source_records_external", "source_slug", "external_id"),
        Index("ix_data_source_records_data_source_id", "data_source_id"),
    )


# ==================== LLM MODELS CACHE ====================


class LLMModelCache(Base):
    """Cached list of available models from each LLM provider.

    Models are fetched from provider APIs and stored here for quick
    lookup in the UI dropdown. Can be refreshed on demand.
    """

    __tablename__ = "llm_model_cache"

    id = Column(String, primary_key=True)
    provider = Column(String, nullable=False)  # openai, anthropic, google, xai, deepseek, ollama, lmstudio
    model_id = Column(String, nullable=False)  # The model identifier used in API calls
    display_name = Column(String, nullable=True)  # Human-readable name
    created_at = Column(DateTime, default=_utcnow)

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
    started_at = Column(DateTime, default=_utcnow)
    completed_at = Column(DateTime, nullable=True)
    duration_seconds = Column(Float, nullable=True)

    entries = relationship("ScratchpadEntry", back_populates="session", cascade="all, delete-orphan")

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
    entry_type = Column(String, nullable=False)  # "thinking", "tool_call", "tool_result", "observation", "answer"
    tool_name = Column(String, nullable=True)  # Which tool was called
    input_data = Column(JSON, nullable=True)  # Tool input or thinking content
    output_data = Column(JSON, nullable=True)  # Tool output or result

    # Token tracking
    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)

    created_at = Column(DateTime, default=_utcnow)

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
    created_at = Column(DateTime, default=_utcnow, nullable=False)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)

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
    created_at = Column(DateTime, default=_utcnow, nullable=False)

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
    clarity_score = Column(Float, nullable=True)  # 0-1: how clear/unambiguous the resolution criteria are
    risk_score = Column(Float, nullable=True)  # 0-1: risk of unexpected resolution
    confidence = Column(Float, nullable=True)  # 0-1: confidence in the analysis

    # Detailed findings
    ambiguities = Column(JSON, nullable=True)  # List of identified ambiguities
    edge_cases = Column(JSON, nullable=True)  # Potential edge cases
    key_dates = Column(JSON, nullable=True)  # Important dates for resolution
    resolution_likelihood = Column(JSON, nullable=True)  # Likelihood assessment per outcome
    summary = Column(Text, nullable=True)  # Human-readable summary
    recommendation = Column(String, nullable=True)  # "safe", "caution", "avoid"

    # Metadata
    session_id = Column(String, ForeignKey("research_sessions.id"), nullable=True)
    model_used = Column(String, nullable=True)
    analyzed_at = Column(DateTime, default=_utcnow)
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
    execution_feasibility = Column(Float, nullable=True)  # Can we execute at these prices?
    market_efficiency = Column(Float, nullable=True)  # Is this a real inefficiency or noise?

    # LLM reasoning
    reasoning = Column(Text, nullable=True)  # Concise decision rationale
    recommendation = Column(String, nullable=False)  # "strong_execute", "execute", "review", "skip", "strong_skip"
    risk_factors = Column(JSON, nullable=True)

    # Comparison with ML classifier
    ml_probability = Column(Float, nullable=True)  # ML classifier's probability
    ml_recommendation = Column(String, nullable=True)  # ML classifier's recommendation
    agreement = Column(Boolean, nullable=True)  # Do ML and LLM agree?

    # Metadata
    session_id = Column(String, ForeignKey("research_sessions.id"), nullable=True)
    model_used = Column(String, nullable=True)
    judged_at = Column(DateTime, default=_utcnow)

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
    started_at = Column(DateTime, default=_utcnow)
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
    provider = Column(String, nullable=False)  # openai, anthropic, google, xai, deepseek, ollama, lmstudio
    model = Column(String, nullable=False)

    # Usage
    input_tokens = Column(Integer, nullable=False)
    output_tokens = Column(Integer, nullable=False)
    cost_usd = Column(Float, nullable=False)

    # Context
    purpose = Column(String, nullable=True)  # "resolution_analysis", "opportunity_judge", etc.
    session_id = Column(String, nullable=True)

    # Timing
    requested_at = Column(DateTime, default=_utcnow)
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
    discovered_at = Column(DateTime, default=_utcnow)
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
    max_drawdown = Column(Float, nullable=True)  # Stored as positive fraction (0.15 = 15% drawdown)
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
    recommendation = Column(String, default="unanalyzed")  # copy_candidate, monitor, avoid, unanalyzed
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

    # Tags (many-to-many via JSON for simplicity)
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

    # Extended metrics (timing skill, execution quality, etc.)
    metrics_json = Column(JSON, nullable=True)

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
    category = Column(String, default="behavioral")  # behavioral, performance, risk, strategy
    color = Column(String, default="#6B7280")  # Hex color for UI
    criteria = Column(JSON, nullable=True)  # Auto-assignment criteria
    created_at = Column(DateTime, default=_utcnow)

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
    detection_method = Column(String, nullable=True)  # funding_source, timing_correlation, pattern_match
    evidence = Column(JSON, nullable=True)

    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)

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
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)

    __table_args__ = (
        Index("idx_trader_group_active", "is_active"),
        Index("idx_trader_group_source", "source_type"),
    )


class TraderGroupMember(Base):
    """Member wallet within a tracked trader group."""

    __tablename__ = "trader_group_members"

    id = Column(String, primary_key=True)
    group_id = Column(String, ForeignKey("trader_groups.id", ondelete="CASCADE"), nullable=False)
    wallet_address = Column(String, nullable=False)
    source = Column(String, default="manual")  # manual, suggested, imported
    confidence = Column(Float, nullable=True)
    notes = Column(Text, nullable=True)
    added_at = Column(DateTime, default=_utcnow)

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
    signal_type = Column(String, nullable=False)  # "multi_wallet_buy", "multi_wallet_sell", "accumulation"
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
    avg_wallet_rank = Column(Float, nullable=True)  # Average rank of participating wallets
    net_notional = Column(Float, nullable=True)
    conflicting_notional = Column(Float, nullable=True)
    market_liquidity = Column(Float, nullable=True)
    market_volume_24h = Column(Float, nullable=True)

    # Status
    is_active = Column(Boolean, default=True)
    first_seen_at = Column(DateTime, default=_utcnow)
    last_seen_at = Column(DateTime, default=_utcnow)
    detected_at = Column(DateTime, default=_utcnow)
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
    created_at = Column(DateTime, default=_utcnow)

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
    cross_platform_arb = Column(Boolean, default=False)  # Trades same event on both platforms
    hedging_detected = Column(Boolean, default=False)
    matching_markets = Column(JSON, default=list)  # Markets traded on both platforms

    confidence = Column(Float, default=0.0)  # Confidence that these are the same entity

    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)

    __table_args__ = (
        Index("idx_cross_platform_poly", "polymarket_address"),
        Index("idx_cross_platform_kalshi", "kalshi_username"),
        Index("idx_cross_platform_pnl", "combined_pnl"),
    )


class LiveTradingRuntimeState(Base):
    """Durable runtime state for the live trading service."""

    __tablename__ = "live_trading_runtime_state"

    id = Column(String, primary_key=True)
    wallet_address = Column(String, nullable=False, unique=True, index=True)
    total_trades = Column(Integer, nullable=False, default=0)
    winning_trades = Column(Integer, nullable=False, default=0)
    losing_trades = Column(Integer, nullable=False, default=0)
    total_volume = Column(Float, nullable=False, default=0.0)
    total_pnl = Column(Float, nullable=False, default=0.0)
    daily_volume = Column(Float, nullable=False, default=0.0)
    daily_pnl = Column(Float, nullable=False, default=0.0)
    open_positions = Column(Integer, nullable=False, default=0)
    last_trade_at = Column(DateTime, nullable=True)
    daily_volume_reset_at = Column(DateTime, nullable=True)
    market_positions_json = Column(JSON, default=dict)
    balance_signature_type = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=_utcnow, nullable=False)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow, nullable=False)

    __table_args__ = (
        Index("idx_live_trading_runtime_wallet", "wallet_address"),
        Index("idx_live_trading_runtime_updated", "updated_at"),
    )


class LiveTradingOrder(Base):
    """Durable order history for live trading."""

    __tablename__ = "live_trading_orders"

    id = Column(String, primary_key=True)
    wallet_address = Column(String, nullable=False, index=True)
    clob_order_id = Column(String, nullable=True, index=True)
    token_id = Column(String, nullable=False, index=True)
    side = Column(String, nullable=False)
    price = Column(Float, nullable=False, default=0.0)
    size = Column(Float, nullable=False, default=0.0)
    order_type = Column(String, nullable=False, default="GTC")
    status = Column(String, nullable=False, default="pending")
    filled_size = Column(Float, nullable=False, default=0.0)
    average_fill_price = Column(Float, nullable=False, default=0.0)
    market_question = Column(Text, nullable=True)
    opportunity_id = Column(String, nullable=True, index=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=_utcnow, nullable=False)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow, nullable=False)

    __table_args__ = (
        Index("idx_live_trading_orders_wallet_created", "wallet_address", "created_at"),
        Index("idx_live_trading_orders_wallet_status", "wallet_address", "status"),
        Index("idx_live_trading_orders_wallet_clob", "wallet_address", "clob_order_id"),
    )


class LiveTradingPosition(Base):
    """Durable snapshot of live positions for restart recovery."""

    __tablename__ = "live_trading_positions"

    id = Column(String, primary_key=True)
    wallet_address = Column(String, nullable=False, index=True)
    token_id = Column(String, nullable=False, index=True)
    market_id = Column(String, nullable=False, index=True)
    market_question = Column(Text, nullable=True)
    outcome = Column(String, nullable=True)
    size = Column(Float, nullable=False, default=0.0)
    average_cost = Column(Float, nullable=False, default=0.0)
    current_price = Column(Float, nullable=False, default=0.0)
    unrealized_pnl = Column(Float, nullable=False, default=0.0)
    created_at = Column(DateTime, default=_utcnow, nullable=False)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow, nullable=False)

    __table_args__ = (
        UniqueConstraint("wallet_address", "token_id", name="uq_live_trading_positions_wallet_token"),
        Index("idx_live_trading_positions_wallet_market", "wallet_address", "market_id"),
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
    started_at = Column(DateTime, default=_utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=False)

    __table_args__ = (
        Index("idx_scanner_runs_completed", "completed_at"),
        Index("idx_scanner_runs_mode", "scan_mode"),
        Index("idx_scanner_runs_success", "success"),
    )


class ScannerBatchQueue(Base):
    """Durable queue of scanner detection batches awaiting aggregation."""

    __tablename__ = "scanner_batch_queue"

    id = Column(String, primary_key=True)
    source = Column(String, nullable=False, default="scanner")
    batch_kind = Column(String, nullable=False, default="scan_cycle")
    opportunities_json = Column(JSON, nullable=False, default=list)
    status_json = Column(JSON, nullable=False, default=dict)
    emitted_at = Column(DateTime, default=_utcnow, nullable=False)
    lease_owner = Column(String, nullable=True)
    lease_expires_at = Column(DateTime, nullable=True)
    attempt_count = Column(Integer, nullable=False, default=0)
    processed_at = Column(DateTime, nullable=True)
    error = Column(Text, nullable=True)

    __table_args__ = (
        Index("idx_scanner_batch_queue_pending", "processed_at", "emitted_at"),
        Index("idx_scanner_batch_queue_emitted", "emitted_at"),
        Index("idx_scanner_batch_queue_lease", "lease_expires_at"),
    )


class StrategyDeadLetterQueue(Base):
    """Durable per-strategy dead-letter queue for failed aggregation groups."""

    __tablename__ = "strategy_dead_letter_queue"

    id = Column(String, primary_key=True)
    source = Column(String, nullable=False, default="scanner")
    batch_id = Column(String, nullable=True)
    strategy_type = Column(String, nullable=False, default="unknown")
    opportunities_json = Column(JSON, nullable=False, default=list)
    status_json = Column(JSON, nullable=False, default=dict)
    first_failed_at = Column(DateTime, default=_utcnow, nullable=False)
    last_failed_at = Column(DateTime, nullable=False, default=_utcnow)
    lease_owner = Column(String, nullable=True)
    lease_expires_at = Column(DateTime, nullable=True)
    attempt_count = Column(Integer, nullable=False, default=0)
    processed_at = Column(DateTime, nullable=True)
    terminal = Column(Boolean, nullable=False, default=False)
    error = Column(Text, nullable=True)

    __table_args__ = (
        Index("idx_strategy_dead_letter_pending", "processed_at", "first_failed_at"),
        Index("idx_strategy_dead_letter_lease", "lease_expires_at"),
        Index("idx_strategy_dead_letter_strategy", "strategy_type", "processed_at"),
    )


class ScannerSloIncident(Base):
    """Durable scanner SLO incident timeline (open/resolved)."""

    __tablename__ = "scanner_slo_incidents"

    id = Column(String, primary_key=True)
    metric = Column(String, nullable=False, index=True)
    severity = Column(String, nullable=False, default="warning")
    status = Column(String, nullable=False, default="open")  # open | resolved
    threshold_value = Column(Float, nullable=True)
    observed_value = Column(Float, nullable=True)
    details_json = Column(JSON, nullable=False, default=dict)
    opened_at = Column(DateTime, default=_utcnow, nullable=False)
    last_seen_at = Column(DateTime, nullable=False, default=_utcnow)
    resolved_at = Column(DateTime, nullable=True)

    __table_args__ = (
        Index("idx_scanner_slo_incidents_status", "status", "opened_at"),
        Index("idx_scanner_slo_incidents_metric_status", "metric", "status"),
        Index("idx_scanner_slo_incidents_last_seen", "last_seen_at"),
    )


class OpportunityState(Base):
    """Current state for each opportunity stable_id (latest known value)."""

    __tablename__ = "opportunity_state"

    stable_id = Column(String, primary_key=True)
    opportunity_json = Column(JSON, nullable=False)
    first_seen_at = Column(DateTime, default=_utcnow, nullable=False)
    last_seen_at = Column(DateTime, nullable=False)
    last_updated_at = Column(DateTime, default=_utcnow, nullable=False)
    is_active = Column(Boolean, nullable=False, default=True)
    last_run_id = Column(String, ForeignKey("scanner_runs.id"), nullable=True)

    __table_args__ = (
        Index("idx_opportunity_state_active", "is_active"),
        Index("idx_opportunity_state_last_seen", "last_seen_at"),
        Index("idx_opportunity_state_last_updated", "last_updated_at"),
        Index("idx_opportunity_state_last_run", "last_run_id"),
    )


class OpportunityEvent(Base):
    """Append-only event log of opportunity lifecycle changes."""

    __tablename__ = "opportunity_events"

    id = Column(String, primary_key=True)
    stable_id = Column(String, nullable=False)
    run_id = Column(String, ForeignKey("scanner_runs.id"), nullable=False)
    event_type = Column(String, nullable=False)  # detected | updated | expired | reactivated
    opportunity_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=_utcnow, nullable=False)

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
    heavy_lane_forced_degraded = Column(Boolean, default=False)
    heavy_lane_degraded_reason = Column(Text, nullable=True)
    heavy_lane_degraded_until = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)


class ScannerSnapshot(Base):
    """Latest scanner output: opportunities + status. Written by scanner worker, read by API."""

    __tablename__ = "scanner_snapshot"

    id = Column(String, primary_key=True, default="latest")
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)
    last_scan_at = Column(DateTime, nullable=True)
    opportunities_json = Column(JSON, default=list)  # list of Opportunity dicts
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


class MarketCatalog(Base):
    """Persisted market catalog from upstream APIs (Polymarket, Kalshi).

    Written by the catalog refresh task; read by scanner on startup and
    as a fallback when the in-memory cache is empty.  Single row with
    id='latest', same pattern as ScannerSnapshot.
    """

    __tablename__ = "market_catalog"

    id = Column(String, primary_key=True, default="latest")
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)
    events_json = Column(JSON, default=list)  # list of Event.model_dump() dicts
    markets_json = Column(JSON, default=list)  # list of Market.model_dump() dicts
    event_count = Column(Integer, default=0)
    market_count = Column(Integer, default=0)
    fetch_duration_seconds = Column(Float, nullable=True)
    error = Column(Text, nullable=True)


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
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)


class NewsWorkflowSnapshot(Base):
    """Latest news workflow output/status written by worker, read by API/UI."""

    __tablename__ = "news_workflow_snapshot"

    id = Column(String, primary_key=True, default="latest")
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)
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
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)


class DiscoverySnapshot(Base):
    """Latest discovery status. Written by discovery worker, read by API/UI."""

    __tablename__ = "discovery_snapshot"

    id = Column(String, primary_key=True, default="latest")
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)
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
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)


class WeatherSnapshot(Base):
    """Latest weather workflow output: opportunities + status."""

    __tablename__ = "weather_snapshot"

    id = Column(String, primary_key=True, default="latest")
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)
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
    status = Column(String, default="pending", nullable=False)  # pending | submitted | executed | skipped | expired
    created_at = Column(DateTime, default=_utcnow, nullable=False)
    consumed_at = Column(DateTime, nullable=True)

    __table_args__ = (
        Index("idx_weather_intent_created", "created_at"),
        Index("idx_weather_intent_status", "status"),
        Index("idx_weather_intent_market", "market_id"),
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
    strategy_context_json = Column(JSON, nullable=True)  # Context from detect() for evaluate()/should_exit()
    quality_passed = Column(Boolean, nullable=True)  # True = passed quality filter at signal creation
    quality_rejection_reasons = Column(JSON, nullable=True)  # List of rejection reason strings
    dedupe_key = Column(String, nullable=False)
    created_at = Column(DateTime, default=_utcnow, nullable=False)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)

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
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)
    stats_json = Column(JSON, default=dict)


# ==================== DB-NATIVE TRADER STRATEGIES ====================


# TraderStrategyDefinition has been removed — all strategies are in the unified
# `strategies` table (Strategy model). The legacy `trader_strategy_definitions`
# table was renamed to `_legacy_trader_strategy_definitions` by migration
# 202602170004 and will be dropped by a future cleanup migration.


class TradeSignalEmission(Base):
    """Immutable history snapshots of signal upserts and status transitions."""

    __tablename__ = "trade_signal_emissions"

    id = Column(String, primary_key=True)
    signal_id = Column(
        String,
        ForeignKey("trade_signals.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    source = Column(String, nullable=False, index=True)
    source_item_id = Column(String, nullable=True)
    signal_type = Column(String, nullable=False)
    strategy_type = Column(String, nullable=True)
    market_id = Column(String, nullable=False, index=True)
    direction = Column(String, nullable=True)
    entry_price = Column(Float, nullable=True)
    effective_price = Column(Float, nullable=True)
    edge_percent = Column(Float, nullable=True)
    confidence = Column(Float, nullable=True)
    liquidity = Column(Float, nullable=True)
    status = Column(String, nullable=False)
    dedupe_key = Column(String, nullable=False)
    event_type = Column(String, nullable=False, index=True)
    reason = Column(Text, nullable=True)
    payload_json = Column(JSON, default=dict)
    snapshot_json = Column(JSON, default=dict)
    created_at = Column(DateTime, default=_utcnow, nullable=False, index=True)

    __table_args__ = (
        Index("idx_trade_signal_emissions_source_created", "source", "created_at"),
        Index("idx_trade_signal_emissions_signal_created", "signal_id", "created_at"),
    )


class ExecutionSimRun(Base):
    """Execution simulator run metadata and aggregate results."""

    __tablename__ = "execution_sim_runs"

    id = Column(String, primary_key=True)
    job_id = Column(String, ForeignKey("validation_jobs.id", ondelete="SET NULL"), nullable=True, index=True)
    strategy_key = Column(String, nullable=False, index=True)
    source_key = Column(String, nullable=False, index=True)
    status = Column(String, nullable=False, default="queued")
    run_seed = Column(String, nullable=True)
    dataset_hash = Column(String, nullable=True)
    config_hash = Column(String, nullable=True)
    code_sha = Column(String, nullable=True)
    market_scope_json = Column(JSON, default=dict)
    params_json = Column(JSON, default=dict)
    requested_start_at = Column(DateTime, nullable=True)
    requested_end_at = Column(DateTime, nullable=True)
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)
    summary_json = Column(JSON, default=dict)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=_utcnow, nullable=False)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)

    __table_args__ = (
        Index("idx_execution_sim_runs_status", "status"),
        Index("idx_execution_sim_runs_created", "created_at"),
    )


class ExecutionSimEvent(Base):
    """Ordered event stream generated by an execution simulator run."""

    __tablename__ = "execution_sim_events"

    id = Column(String, primary_key=True)
    run_id = Column(
        String,
        ForeignKey("execution_sim_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    sequence = Column(Integer, nullable=False)
    event_type = Column(String, nullable=False, index=True)
    event_at = Column(DateTime, nullable=False, index=True)
    signal_id = Column(String, nullable=True, index=True)
    market_id = Column(String, nullable=True, index=True)
    direction = Column(String, nullable=True)
    price = Column(Float, nullable=True)
    quantity = Column(Float, nullable=True)
    notional_usd = Column(Float, nullable=True)
    fees_usd = Column(Float, nullable=True)
    slippage_bps = Column(Float, nullable=True)
    realized_pnl_usd = Column(Float, nullable=True)
    unrealized_pnl_usd = Column(Float, nullable=True)
    payload_json = Column(JSON, default=dict)
    created_at = Column(DateTime, default=_utcnow, nullable=False)

    __table_args__ = (
        UniqueConstraint("run_id", "sequence", name="uq_execution_sim_events_run_sequence"),
        Index("idx_execution_sim_events_run_event_at", "run_id", "event_at"),
    )


# ==================== WORKER RUNTIME STATE ====================


class WorkerControl(Base):
    """Generic worker control row for independently owned worker loops."""

    __tablename__ = "worker_control"

    worker_name = Column(String, primary_key=True)
    is_enabled = Column(Boolean, default=True)
    is_paused = Column(Boolean, default=False)
    interval_seconds = Column(Integer, default=60)
    requested_run_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)


class WorkerSnapshot(Base):
    """Latest worker status snapshot for API/websocket health surfaces."""

    __tablename__ = "worker_snapshot"

    worker_name = Column(String, primary_key=True)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)
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
    mode = Column(String, default="shadow")
    run_interval_seconds = Column(Integer, default=5)
    requested_run_at = Column(DateTime, nullable=True)
    kill_switch = Column(Boolean, default=False)
    settings_json = Column(JSON, default=dict)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)


class TraderOrchestratorSnapshot(Base):
    """Latest orchestrator status/performance snapshot."""

    __tablename__ = "trader_orchestrator_snapshot"

    id = Column(String, primary_key=True, default="latest")
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)
    last_run_at = Column(DateTime, nullable=True)
    running = Column(Boolean, default=False)
    enabled = Column(Boolean, default=False)
    current_activity = Column(String, nullable=True)
    interval_seconds = Column(Integer, default=5)
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
    source_configs_json = Column(JSON, default=list)
    risk_limits_json = Column(JSON, default=dict)
    metadata_json = Column(JSON, default=dict)
    mode = Column(String, nullable=False, default="shadow")
    is_enabled = Column(Boolean, default=True)
    is_paused = Column(Boolean, default=False)
    interval_seconds = Column(Integer, default=60)
    requested_run_at = Column(DateTime, nullable=True)
    last_run_at = Column(DateTime, nullable=True)
    next_run_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=_utcnow, nullable=False)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)


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
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)


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
    strategy_version = Column(Integer, nullable=True, index=True)
    decision = Column(String, nullable=False)  # selected | skipped | blocked | failed
    reason = Column(Text, nullable=True)
    score = Column(Float, nullable=True)
    event_id = Column(String, nullable=True, index=True)
    trace_id = Column(String, nullable=True, index=True)
    checks_summary_json = Column(JSON, default=dict)
    risk_snapshot_json = Column(JSON, default=dict)
    payload_json = Column(JSON, default=dict)
    created_at = Column(DateTime, default=_utcnow, nullable=False)

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
    created_at = Column(DateTime, default=_utcnow, nullable=False)

    __table_args__ = (Index("idx_trader_decision_checks_decision_created", "decision_id", "created_at"),)


class TraderOrder(Base):
    """Execution records owned by a trader and tied to a decision/signal."""

    __tablename__ = "trader_orders"

    id = Column(String, primary_key=True)
    trader_id = Column(String, nullable=False, index=True)
    signal_id = Column(String, ForeignKey("trade_signals.id"), nullable=True, index=True)
    decision_id = Column(
        String,
        ForeignKey("trader_decisions.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    source = Column(String, nullable=False, index=True)
    strategy_key = Column(String, nullable=True, index=True)
    strategy_version = Column(Integer, nullable=True, index=True)
    market_id = Column(String, nullable=False, index=True)
    market_question = Column(Text, nullable=True)
    direction = Column(String, nullable=True)
    event_id = Column(String, nullable=True, index=True)
    trace_id = Column(String, nullable=True, index=True)
    mode = Column(String, nullable=False, default="shadow")
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
    created_at = Column(DateTime, default=_utcnow, nullable=False)
    executed_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)

    __table_args__ = (
        Index("idx_trader_orders_created", "created_at"),
        Index("idx_trader_orders_status", "status"),
        Index("idx_trader_orders_trader_created", "trader_id", "created_at"),
    )


class ExecutionSession(Base):
    """Persistent multi-leg execution session owned by the trader orchestrator."""

    __tablename__ = "execution_sessions"

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
    decision_id = Column(
        String,
        ForeignKey("trader_decisions.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    source = Column(String, nullable=False, index=True)
    strategy_key = Column(String, nullable=True, index=True)
    strategy_version = Column(Integer, nullable=True, index=True)
    mode = Column(String, nullable=False, default="shadow")
    status = Column(String, nullable=False, default="pending")
    policy = Column(String, nullable=True)
    plan_id = Column(String, nullable=True, index=True)
    market_ids_json = Column(JSON, default=list)
    legs_total = Column(Integer, nullable=False, default=0)
    legs_completed = Column(Integer, nullable=False, default=0)
    legs_failed = Column(Integer, nullable=False, default=0)
    legs_open = Column(Integer, nullable=False, default=0)
    requested_notional_usd = Column(Float, nullable=True)
    executed_notional_usd = Column(Float, nullable=False, default=0.0)
    max_unhedged_notional_usd = Column(Float, nullable=False, default=0.0)
    unhedged_notional_usd = Column(Float, nullable=False, default=0.0)
    trace_id = Column(String, nullable=True, index=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    expires_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    payload_json = Column(JSON, default=dict)
    created_at = Column(DateTime, default=_utcnow, nullable=False)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)

    __table_args__ = (
        Index("idx_execution_sessions_created", "created_at"),
        Index("idx_execution_sessions_status", "status"),
        Index("idx_execution_sessions_trader_status", "trader_id", "status"),
    )


class ExecutionSessionLeg(Base):
    """Plan leg state for an execution session."""

    __tablename__ = "execution_session_legs"

    id = Column(String, primary_key=True)
    session_id = Column(
        String,
        ForeignKey("execution_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    leg_index = Column(Integer, nullable=False)
    leg_id = Column(String, nullable=False)
    market_id = Column(String, nullable=False, index=True)
    market_question = Column(Text, nullable=True)
    token_id = Column(String, nullable=True)
    side = Column(String, nullable=False, default="buy")
    outcome = Column(String, nullable=True)
    price_policy = Column(String, nullable=False, default="maker_limit")
    time_in_force = Column(String, nullable=False, default="GTC")
    post_only = Column(Boolean, nullable=False, default=False, server_default="false")
    target_price = Column(Float, nullable=True)
    requested_notional_usd = Column(Float, nullable=True)
    requested_shares = Column(Float, nullable=True)
    filled_notional_usd = Column(Float, nullable=False, default=0.0)
    filled_shares = Column(Float, nullable=False, default=0.0)
    avg_fill_price = Column(Float, nullable=True)
    status = Column(String, nullable=False, default="pending")
    last_error = Column(Text, nullable=True)
    metadata_json = Column(JSON, default=dict)
    created_at = Column(DateTime, default=_utcnow, nullable=False)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)

    __table_args__ = (
        UniqueConstraint("session_id", "leg_index", name="uq_execution_session_leg_index"),
        Index("idx_execution_session_legs_session_status", "session_id", "status"),
    )


class ExecutionSessionOrder(Base):
    """Order-level activity emitted while a session leg is being worked."""

    __tablename__ = "execution_session_orders"

    id = Column(String, primary_key=True)
    session_id = Column(
        String,
        ForeignKey("execution_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    leg_id = Column(
        String,
        ForeignKey("execution_session_legs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    trader_order_id = Column(
        String,
        ForeignKey("trader_orders.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    provider_order_id = Column(String, nullable=True, index=True)
    provider_clob_order_id = Column(String, nullable=True, index=True)
    action = Column(String, nullable=False, default="submit")
    side = Column(String, nullable=False)
    price = Column(Float, nullable=True)
    size = Column(Float, nullable=True)
    notional_usd = Column(Float, nullable=True)
    status = Column(String, nullable=False, default="submitted")
    reason = Column(Text, nullable=True)
    payload_json = Column(JSON, default=dict)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=_utcnow, nullable=False)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)

    __table_args__ = (
        Index("idx_execution_session_orders_session_created", "session_id", "created_at"),
        Index("idx_execution_session_orders_leg_created", "leg_id", "created_at"),
    )


class ExecutionSessionEvent(Base):
    """Immutable event stream for session state transitions and policy actions."""

    __tablename__ = "execution_session_events"

    id = Column(String, primary_key=True)
    session_id = Column(
        String,
        ForeignKey("execution_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    leg_id = Column(
        String,
        ForeignKey("execution_session_legs.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    event_type = Column(String, nullable=False, index=True)
    severity = Column(String, nullable=False, default="info")
    message = Column(Text, nullable=True)
    payload_json = Column(JSON, default=dict)
    created_at = Column(DateTime, default=_utcnow, nullable=False, index=True)

    __table_args__ = (Index("idx_execution_session_events_session_created", "session_id", "created_at"),)


class TraderPosition(Base):
    """Aggregated position inventory per trader/market/direction/mode."""

    __tablename__ = "trader_positions"

    id = Column(String, primary_key=True)
    trader_id = Column(
        String,
        ForeignKey("traders.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    mode = Column(String, nullable=False, default="shadow")
    market_id = Column(String, nullable=False, index=True)
    market_question = Column(Text, nullable=True)
    direction = Column(String, nullable=True)
    status = Column(String, nullable=False, default="open")  # open | closed
    open_order_count = Column(Integer, nullable=False, default=0)
    total_notional_usd = Column(Float, nullable=False, default=0.0)
    avg_entry_price = Column(Float, nullable=True)
    first_order_at = Column(DateTime, nullable=True)
    last_order_at = Column(DateTime, nullable=True)
    closed_at = Column(DateTime, nullable=True)
    payload_json = Column(JSON, default=dict)
    created_at = Column(DateTime, default=_utcnow, nullable=False)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)

    __table_args__ = (
        UniqueConstraint(
            "trader_id",
            "mode",
            "market_id",
            "direction",
            name="uq_trader_position_identity",
        ),
        Index("idx_trader_positions_status", "status"),
        Index("idx_trader_positions_trader_status", "trader_id", "status"),
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
    consumed_at = Column(DateTime, default=_utcnow, nullable=False)

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
    created_at = Column(DateTime, default=_utcnow, nullable=False, index=True)

    __table_args__ = (Index("idx_trader_events_type_created", "event_type", "created_at"),)


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
    created_at = Column(DateTime, default=_utcnow, nullable=False, index=True)


class StrategyExperiment(Base):
    """Strategy version A/B experiment configuration."""

    __tablename__ = "strategy_experiments"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    source_key = Column(String, nullable=False, index=True)
    strategy_key = Column(String, nullable=False, index=True)
    control_version = Column(Integer, nullable=False)
    candidate_version = Column(Integer, nullable=False)
    candidate_allocation_pct = Column(Float, nullable=False, default=50.0)
    scope_json = Column(JSON, default=dict)
    status = Column(String, nullable=False, default="active")  # active | paused | completed | archived
    created_by = Column(String, nullable=True)
    notes = Column(Text, nullable=True)
    metadata_json = Column(JSON, default=dict)
    promoted_version = Column(Integer, nullable=True)
    ended_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=_utcnow, nullable=False)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)

    __table_args__ = (
        Index("idx_strategy_experiments_strategy_status", "strategy_key", "status"),
        Index("idx_strategy_experiments_source_status", "source_key", "status"),
        Index("idx_strategy_experiments_created", "created_at"),
    )


class StrategyExperimentAssignment(Base):
    """Signal-level deterministic assignment for active strategy experiments."""

    __tablename__ = "strategy_experiment_assignments"

    id = Column(String, primary_key=True)
    experiment_id = Column(
        String,
        ForeignKey("strategy_experiments.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    trader_id = Column(
        String,
        ForeignKey("traders.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    signal_id = Column(
        String,
        ForeignKey("trade_signals.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    decision_id = Column(
        String,
        ForeignKey("trader_decisions.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    order_id = Column(
        String,
        ForeignKey("trader_orders.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    source_key = Column(String, nullable=False, index=True)
    strategy_key = Column(String, nullable=False, index=True)
    strategy_version = Column(Integer, nullable=False, index=True)
    assignment_group = Column(String, nullable=False)  # control | candidate
    payload_json = Column(JSON, default=dict)
    created_at = Column(DateTime, default=_utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)

    __table_args__ = (
        UniqueConstraint(
            "experiment_id",
            "trader_id",
            "signal_id",
            name="uq_strategy_experiment_assignment_signal",
        ),
        Index("idx_strategy_experiment_assignments_group", "experiment_id", "assignment_group"),
    )


# ==================== EVENTS ====================


class EventsSignal(Base):
    """Aggregated events signal from any source."""

    __tablename__ = "events_signals"

    id = Column(String, primary_key=True)
    signal_type = Column(
        String, nullable=False
    )  # conflict, tension, instability, convergence, anomaly, military, infrastructure
    severity = Column(Float, nullable=False, default=0.0)  # 0-1 normalized
    country = Column(String, nullable=True)
    iso3 = Column(String, nullable=True)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    title = Column(Text, nullable=False)
    description = Column(Text, nullable=True)
    source = Column(String, nullable=True)
    detected_at = Column(DateTime, default=_utcnow, nullable=False)
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


class EventsSnapshot(Base):
    """Worker snapshot for events collector."""

    __tablename__ = "events_snapshots"

    id = Column(String, primary_key=True, default="latest")
    status = Column(JSON, nullable=True)
    signals_json = Column(JSON, nullable=True)  # last batch of signals
    stats = Column(JSON, nullable=True)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)


# ==================== DATABASE SETUP ====================

_engine_kw: dict = {
    "echo": False,
    "pool_pre_ping": True,
}

_database_url = str(settings.DATABASE_URL or "").strip().lower()
if not _database_url.startswith("postgresql"):
    raise ValueError(f"DATABASE_URL must use PostgreSQL; got {settings.DATABASE_URL!r}")

# Worker subprocesses need much smaller pools.  The main backend handles
# API requests, WebSockets, and in-process services that benefit from a
# larger pool.  Each of the 14 worker processes only runs one loop and
# rarely needs more than 1-2 concurrent connections.  Using the full
# pool_size=20 per worker would demand 15×50=750 connections against
# Postgres (default max_connections=100), causing connection refusals
# that crash the entire system.
_is_worker = _os.environ.get("HOMERUN_PROCESS_ROLE") == "worker"
if _is_worker:
    _pool_size = max(1, int(settings.DATABASE_WORKER_POOL_SIZE))
    _max_overflow = max(0, int(settings.DATABASE_WORKER_MAX_OVERFLOW))
else:
    _pool_size = max(1, int(settings.DATABASE_POOL_SIZE))
    _max_overflow = max(0, int(settings.DATABASE_MAX_OVERFLOW))

_engine_kw.update(
    {
        "pool_size": _pool_size,
        "max_overflow": _max_overflow,
        "pool_timeout": max(1, int(settings.DATABASE_POOL_TIMEOUT_SECONDS)),
        "pool_recycle": max(30, int(settings.DATABASE_POOL_RECYCLE_SECONDS)),
        "pool_use_lifo": True,
    }
)
_engine_kw["connect_args"] = {
    "timeout": float(max(1.0, float(settings.DATABASE_CONNECT_TIMEOUT_SECONDS))),
    "command_timeout": float(
        max(
            5.0,
            float(settings.DATABASE_POOL_TIMEOUT_SECONDS),
            (float(settings.DATABASE_STATEMENT_TIMEOUT_MS) / 1000.0) + 5.0,
        )
    ),
    "server_settings": {
        "timezone": "UTC",
        "statement_timeout": str(max(1000, int(settings.DATABASE_STATEMENT_TIMEOUT_MS))),
        "idle_in_transaction_session_timeout": str(max(1000, int(settings.DATABASE_IDLE_IN_TRANSACTION_TIMEOUT_MS))),
    },
}

async_engine = create_async_engine(settings.DATABASE_URL, **_engine_kw)

_db_logger = _logging.getLogger("homerun.db.pool")
_db_logger.info(
    "Connection pool created (role=%s, pool_size=%d, max_overflow=%d)",
    "worker" if _is_worker else "main",
    _pool_size,
    _max_overflow,
)


def _pool_task_context() -> tuple[str, str]:
    try:
        task = asyncio.current_task()
    except RuntimeError:
        return "sync", "sync"
    if task is None:
        return "unknown", "unknown"
    task_name = "unknown"
    try:
        task_name = task.get_name() or f"task-{id(task)}"
    except Exception:
        task_name = f"task-{id(task)}"
    coro = task.get_coro()
    coro_name = getattr(coro, "__qualname__", getattr(coro, "__name__", type(coro).__name__))
    return task_name, str(coro_name or "unknown")


@_sa_event.listens_for(async_engine.sync_engine, "checkout")
def _on_checkout(dbapi_connection, connection_record, connection_proxy):
    task_name, coro_name = _pool_task_context()
    connection_record.info["checkout_time"] = _time.monotonic()
    connection_record.info["checkout_task_name"] = task_name
    connection_record.info["checkout_task_coro"] = coro_name

@_sa_event.listens_for(async_engine.sync_engine, "checkin")
def _on_checkin(dbapi_connection, connection_record):
    checkout_time = connection_record.info.pop("checkout_time", None)
    checkout_task_name = connection_record.info.pop("checkout_task_name", "unknown")
    checkout_task_coro = connection_record.info.pop("checkout_task_coro", "unknown")
    if checkout_time is not None:
        elapsed = _time.monotonic() - checkout_time
        if elapsed > 15.0:
            _db_logger.warning(
                "Connection held for %.1fs before return to pool (task=%s, coro=%s)",
                elapsed,
                checkout_task_name,
                checkout_task_coro,
            )

@_sa_event.listens_for(async_engine.sync_engine, "invalidate")
def _on_invalidate(dbapi_connection, connection_record, exception):
    checkout_time = connection_record.info.get("checkout_time")
    checkout_task_name = connection_record.info.get("checkout_task_name", "unknown")
    checkout_task_coro = connection_record.info.get("checkout_task_coro", "unknown")
    exception_name = type(exception).__name__ if exception else "None"
    if exception_name == "CancelledError":
        if checkout_time is not None:
            elapsed = _time.monotonic() - checkout_time
            _db_logger.debug(
                "Connection invalidated after %.1fs checked out due to cancellation (task=%s, coro=%s)",
                elapsed,
                checkout_task_name,
                checkout_task_coro,
            )
        else:
            _db_logger.debug(
                "Connection invalidated due to cancellation (task=%s, coro=%s)",
                checkout_task_name,
                checkout_task_coro,
            )
        return
    if checkout_time is not None:
        elapsed = _time.monotonic() - checkout_time
        _db_logger.warning(
            "Connection invalidated after %.1fs checked out (exception=%s, task=%s, coro=%s)",
            elapsed,
            exception_name,
            checkout_task_name,
            checkout_task_coro,
        )
        return
    _db_logger.warning(
        "Connection invalidated (exception=%s, task=%s, coro=%s)",
        exception_name,
        checkout_task_name,
        checkout_task_coro,
    )

AsyncSessionLocal = sessionmaker(async_engine, class_=RetryableAsyncSession, expire_on_commit=False)


async def recover_pool() -> None:
    """Drop all pooled connections and force fresh ones on next checkout.

    Call after consecutive DB disconnect errors to clear stale connections
    that pool_pre_ping cannot reclaim fast enough during mass-disconnect
    scenarios (e.g. PostgreSQL restart).
    """
    await async_engine.dispose()


# ==================== CONNECTION CHECKOUT REAPER ====================
#
# Runs as a background task and periodically walks every connection in the
# pool.  Any connection that has been checked out for longer than the hard
# limit is forcibly invalidated — the underlying TCP socket is closed and
# the connection_record is returned to the pool so a fresh connection can
# be created on the next checkout.
#
# This prevents a single stuck task (e.g. an external HTTP call that hangs)
# from permanently consuming a pool slot and eventually exhausting the
# pool, which would take down the entire application.

_CHECKOUT_HARD_LIMIT_SECONDS = 120.0  # 2 minutes — no single operation should ever take this long
_REAPER_SCAN_INTERVAL_SECONDS = 10.0
_POOL_EXHAUSTION_THRESHOLD = 0.85     # trigger recovery when 85% of slots are checked out
_reaper_task: asyncio.Task | None = None


def _get_pool_stats() -> dict:
    """Return current pool utilization stats."""
    pool = async_engine.pool
    checked_in = pool.checkedin()
    checked_out = pool.checkedout()
    overflow = pool.overflow()
    pool_size = pool.size()
    total_capacity = pool_size + _max_overflow
    return {
        "checked_in": checked_in,
        "checked_out": checked_out,
        "overflow": overflow,
        "pool_size": pool_size,
        "max_overflow": _max_overflow,
        "total_capacity": total_capacity,
        "utilization": checked_out / total_capacity if total_capacity > 0 else 0.0,
    }


async def _reap_stale_checkouts() -> None:
    """Walk the pool and invalidate connections held beyond the hard limit."""
    pool = async_engine.pool
    now = _time.monotonic()
    reaped = 0

    # SQLAlchemy's QueuePool tracks connection records internally.
    # _pool is the queue of idle connections, but checked-out connections
    # live on _all_conns (the full set of managed connection_records).
    all_records = getattr(pool, "_all_conns", None)
    if all_records is None:
        return

    for conn_record in list(all_records):
        info = getattr(conn_record, "info", None)
        if info is None:
            continue
        checkout_time = info.get("checkout_time")
        if checkout_time is None:
            continue  # checked in, not out
        elapsed = now - checkout_time
        if elapsed < _CHECKOUT_HARD_LIMIT_SECONDS:
            continue

        task_name = info.get("checkout_task_name", "unknown")
        coro_name = info.get("checkout_task_coro", "unknown")
        _db_logger.error(
            "REAPER: Force-invalidating connection held for %.1fs (limit=%.0fs, task=%s, coro=%s)",
            elapsed,
            _CHECKOUT_HARD_LIMIT_SECONDS,
            task_name,
            coro_name,
        )
        # Invalidate the underlying DBAPI connection.  This closes the TCP
        # socket, which causes any in-flight query to fail with a connection
        # error.  The connection_record is returned to the pool and will
        # create a fresh connection on next checkout.
        try:
            dbapi_conn = conn_record.dbapi_connection
            if dbapi_conn is not None:
                conn_record.invalidate(e=TimeoutError(f"checkout exceeded {_CHECKOUT_HARD_LIMIT_SECONDS}s"))
                reaped += 1
        except Exception as e:
            _db_logger.warning("REAPER: Failed to invalidate connection: %s", e)

    return reaped


async def _pool_watchdog_loop() -> None:
    """Background loop: reap stale checkouts and recover exhausted pools."""
    _db_logger.info(
        "Pool watchdog started (hard_limit=%.0fs, scan_interval=%.0fs, exhaustion_threshold=%.0f%%)",
        _CHECKOUT_HARD_LIMIT_SECONDS,
        _REAPER_SCAN_INTERVAL_SECONDS,
        _POOL_EXHAUSTION_THRESHOLD * 100,
    )
    consecutive_exhausted = 0

    while True:
        try:
            await asyncio.sleep(_REAPER_SCAN_INTERVAL_SECONDS)

            # Phase 1: reap any connections held too long
            reaped = await _reap_stale_checkouts()
            if reaped:
                _db_logger.warning("REAPER: Invalidated %d stale connection(s)", reaped)

            # Phase 2: check pool utilization
            stats = _get_pool_stats()
            if stats["utilization"] >= _POOL_EXHAUSTION_THRESHOLD:
                consecutive_exhausted += 1
                _db_logger.error(
                    "Pool near exhaustion: %d/%d checked out (%.0f%%), streak=%d",
                    stats["checked_out"],
                    stats["total_capacity"],
                    stats["utilization"] * 100,
                    consecutive_exhausted,
                )
                # After 3 consecutive scans at near-exhaustion, force pool
                # recovery — dispose all connections and start fresh.
                if consecutive_exhausted >= 3:
                    _db_logger.error(
                        "POOL RECOVERY: Disposing all connections after %d consecutive exhaustion scans",
                        consecutive_exhausted,
                    )
                    await async_engine.dispose()
                    consecutive_exhausted = 0
            else:
                consecutive_exhausted = 0

        except asyncio.CancelledError:
            _db_logger.info("Pool watchdog stopped")
            return
        except Exception as e:
            _db_logger.error("Pool watchdog error: %s", e, exc_info=True)


def start_pool_watchdog() -> asyncio.Task:
    """Start the pool watchdog background task.  Safe to call multiple times."""
    global _reaper_task
    if _reaper_task is not None and not _reaper_task.done():
        return _reaper_task
    _reaper_task = asyncio.create_task(_pool_watchdog_loop(), name="pool-watchdog")
    return _reaper_task


def stop_pool_watchdog() -> None:
    """Cancel the pool watchdog task."""
    global _reaper_task
    if _reaper_task is not None and not _reaper_task.done():
        _reaper_task.cancel()
    _reaper_task = None


# ==================== BOUNDED SESSION CONTEXT MANAGER ====================
#
# Wraps AsyncSessionLocal with an overall timeout.  If the timeout fires,
# the session is rolled back and closed, and asyncio.TimeoutError is raised.
# This prevents any single database operation from holding a connection
# indefinitely, even if the work involves awaiting external I/O while a
# session is open.

from contextlib import asynccontextmanager as _asynccontextmanager


@_asynccontextmanager
async def release_conn(session):
    """Temporarily return the session's DB connection to the pool.

    Use this around external HTTP calls that do not need the database so
    that the connection is not held hostage while waiting on network I/O::

        async with release_conn(session):
            result = await some_http_call()   # no DB conn held

    On entry the session is flushed, the underlying connection is returned
    to the pool, and the session is reset.  On exit the next DB operation
    on the session transparently checks out a fresh connection.

    Any unflushed ORM state is lost (flush is called first to persist it).
    Callers should commit before entering if they need durability.
    """
    if not session.is_active:
        yield
        return
    try:
        if session.dirty or session.new:
            await session.flush()
    except Exception:
        pass
    try:
        await session.reset()
    except Exception:
        pass
    try:
        yield
    finally:
        # No action needed — the session lazily reconnects on next use.
        pass


@_asynccontextmanager
async def bounded_session(timeout_seconds: float = 30.0):
    """Acquire a database session with a hard overall time limit.

    Usage::

        async with bounded_session(timeout_seconds=15.0) as session:
            result = await session.execute(...)
            await session.commit()

    If the block takes longer than *timeout_seconds*, the session is
    rolled back and closed, and ``asyncio.TimeoutError`` is raised.
    """
    session = AsyncSessionLocal()
    deadline = _time.monotonic() + timeout_seconds
    try:
        yield session
    except asyncio.TimeoutError:
        try:
            await session.rollback()
        except Exception:
            pass
        raise
    except Exception:
        try:
            await session.rollback()
        except Exception:
            pass
        raise
    finally:
        remaining = deadline - _time.monotonic()
        if remaining <= 0:
            _db_logger.warning(
                "bounded_session exceeded %.1fs deadline, forcing close",
                timeout_seconds,
            )
        await session.close()


def _run_alembic_upgrade(connection) -> None:
    from alembic import command
    from alembic.config import Config

    # Temporarily remove statement_timeout during migrations so that
    # long-running backfill / DDL statements are not killed mid-flight.
    from sqlalchemy import text as _text
    connection.execute(_text("SET LOCAL statement_timeout = 0"))
    connection.execute(_text("SET LOCAL idle_in_transaction_session_timeout = 0"))

    backend_root = Path(__file__).resolve().parents[1]
    alembic_cfg = Config(str(backend_root / "alembic.ini"))
    alembic_cfg.set_main_option("script_location", str(backend_root / "alembic"))
    alembic_cfg.set_main_option("sqlalchemy.url", str(connection.engine.url))
    alembic_cfg.attributes["connection"] = connection
    command.upgrade(alembic_cfg, "head")


async def init_database():
    """Initialize database and apply Alembic migrations."""
    from models.model_registry import register_all_models

    register_all_models()
    async with async_engine.begin() as conn:
        await conn.run_sync(_run_alembic_upgrade)


async def get_db_session() -> AsyncSession:
    """Get database session via FastAPI ``Depends()``."""
    session = AsyncSessionLocal()
    try:
        yield session
    except BaseException:
        if session.in_transaction():
            try:
                await session.rollback()
            except Exception:
                try:
                    await session.invalidate()
                except Exception:
                    pass
        raise
    finally:
        # close() is guaranteed to never raise (it handles
        # CancelledError and falls back to invalidate internally).
        await session.close()
