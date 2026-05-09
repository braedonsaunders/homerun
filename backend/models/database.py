from sqlalchemy import (
    Column,
    String,
    Integer,
    BigInteger,
    Boolean,
    Text,
    JSON,
    ForeignKey,
    Enum as SQLEnum,
    Index,
    UniqueConstraint,
    event as _sa_event,
    text,
)
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.exc import DBAPIError
from sqlalchemy.types import TypeDecorator, DateTime as SADateTime
from datetime import datetime, timezone
from pathlib import Path
from contextlib import asynccontextmanager as _asynccontextmanager
import logging as _logging
import enum
import asyncio
import os as _os
import time as _time
import warnings as _warnings
import weakref
from config import settings
from models.types import PreciseFloat as Float

# SAWarning fired from AsyncAdaptedQueuePool._finalize_fairy when the
# *connection's* __del__ runs before the session's __del__ has finished
# scheduling cleanup.  RetryableAsyncSession.__del__ already schedules
# an async _do_close_or_invalidate, so the underlying connection is
# either properly closed or invalidated — the pool's GC complaint is
# stale by the time it logs.  Filter the specific message.
_warnings.filterwarnings(
    "ignore",
    message=r"The garbage collector is trying to clean up non-checked-in connection.*",
    category=Warning,
)


class _DropFinalizeFairyGCWarning(_logging.Filter):
    """Drop the GC-cleanup error from sqlalchemy's pool finalizer.

    Same root cause as the SAWarning above — the connection's __del__
    races our session-side cleanup.  The pool log call is on a separate
    channel from warnings.filterwarnings, so we silence it here too.
    """

    def filter(self, record: _logging.LogRecord) -> bool:  # type: ignore[override]
        if "garbage collector is trying to clean up non-checked-in connection" in record.getMessage():
            return False
        return True


_logging.getLogger("sqlalchemy.pool.impl.AsyncAdaptedQueuePool").addFilter(_DropFinalizeFairyGCWarning())
_logging.getLogger("sqlalchemy.pool").addFilter(_DropFinalizeFairyGCWarning())

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
        # Driver normally returns ``datetime``; defensively handle the
        # case where a value somehow arrives as a string (e.g. raw SQL
        # result, JSON column re-typed, asyncpg fallback path).  The
        # previous unguarded ``value.tzinfo`` access raised
        # ``AttributeError: 'str' object has no attribute 'tzinfo'``
        # which surfaced in production as
        # ``Worker freshness check failed plane=all`` every 30s and
        # triggered spurious worker restarts.
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc)
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            try:
                parsed = datetime.fromisoformat(text)
            except Exception:
                return None
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        return None


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

    # Per-session in-flight inner tasks. execute/commit/flush wrap the
    # underlying SQLAlchemy call as a Task and shield it; if the calling
    # task is cancelled, the inner Task keeps running until the asyncpg
    # extended-protocol exchange completes.  close() must wait for these
    # before releasing the connection — otherwise super().close() races
    # with the still-active inner task and asyncpg raises
    # ``cannot perform operation: another operation is in progress``.
    _DRAIN_TIMEOUT_SECONDS = 10.0

    def _get_inflight(self) -> set:
        bag = getattr(self, "_inflight_tasks", None)
        if bag is None:
            bag = set()
            setattr(self, "_inflight_tasks", bag)
        return bag

    def _track_inflight(self, task) -> None:
        bag = self._get_inflight()
        bag.add(task)
        task.add_done_callback(bag.discard)

    async def _wait_inflight(self) -> None:
        """Drain any prior shielded inner tasks before starting new work.

        If a previous execute/commit/flush was cancelled by ``wait_for``,
        its shielded inner task continues running until the asyncpg
        protocol exchange completes.  Starting a new operation on the
        same session before that inner finishes raises
        ``This session is provisioning a new connection; concurrent
        operations are not permitted`` (SQLAlchemy isce) or
        ``cannot perform operation: another operation is in progress``
        (asyncpg).  Calling this at the top of every public method that
        touches the connection serializes the cleanup with new work.
        """
        bag = getattr(self, "_inflight_tasks", None)
        if not bag:
            return
        pending = [t for t in list(bag) if not t.done()]
        if not pending:
            return
        try:
            await asyncio.wait(pending, timeout=self._DRAIN_TIMEOUT_SECONDS)
        except Exception:
            pass

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
        # Wait for any shielded inner tasks left by a cancelled
        # execute/commit/flush before starting rollback — otherwise
        # rollback's connection use races the leftover protocol and
        # raises isce or "another operation in progress".
        await self._wait_inflight()
        try:
            await super().rollback()
        except asyncio.CancelledError:
            # On cancellation, invalidate immediately instead of trying
            # a fire-and-forget rollback that races with __aexit__'s
            # close on the same connection. See execute() docstring.
            self._fire_and_forget(self._do_invalidate())
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

    async def commit(self) -> None:
        """Cancellation-safe commit with retry on transient errors.

        asyncpg sends the commit over the extended protocol as a
        Parse/Bind/Execute/Sync sequence.  If a CancelledError hits
        mid-sequence the server is left in state=active with
        wait_event=Client/ClientRead and an open transaction that
        nothing else can reap until the worker restarts.  Shielding
        the commit lets the whole sequence finish atomically while
        still re-raising CancelledError to the caller afterwards.

        Connection-broken errors (e.g. asyncpg
        ``cannot switch to state X; another operation in progress``)
        are NOT retried at the session level — the transaction's
        writes are already lost and retrying ``commit()`` on a
        rolled-back session would silently report success without
        persisting anything.  The session is invalidated so the pool
        drops the poisoned connection, and the error propagates so
        the caller can either re-issue the writes or abandon the
        cycle.
        """
        from utils.retry import is_db_connection_broken

        await self._wait_inflight()
        for attempt in range(1, self._COMMIT_RETRY_ATTEMPTS + 1):
            inner = asyncio.ensure_future(super().commit())
            self._track_inflight(inner)
            try:
                await asyncio.shield(inner)
                return
            except asyncio.CancelledError:
                # Drain the inner commit before any cleanup so the
                # extended-protocol sequence finishes atomically.
                self._fire_and_forget(self._drain_then_invalidate(inner))
                raise
            except DBAPIError as exc:
                if is_db_connection_broken(exc):
                    self._fire_and_forget(self._drain_then_invalidate(inner))
                    raise
                message = str(getattr(exc, "orig", exc)).lower()
                retryable = any(marker in message for marker in self._COMMIT_RETRYABLE_MESSAGES)
                if not retryable or attempt >= self._COMMIT_RETRY_ATTEMPTS:
                    raise
                await self._reset_after_failed_commit()
                delay = min(self._COMMIT_BASE_DELAY_SECONDS * (2 ** (attempt - 1)), 0.4)
                await asyncio.sleep(delay)
            except Exception as exc:
                # Same rationale as ``execute()``: catch raw driver-level
                # connection-broken errors that aren't wrapped as
                # ``DBAPIError`` (asyncpg ``InternalClientError`` etc.)
                # and drain-then-invalidate before re-raising, so the
                # poisoned connection doesn't go back to the pool.
                if is_db_connection_broken(exc):
                    self._fire_and_forget(self._drain_then_invalidate(inner))
                else:
                    self._fire_and_forget(self._do_rollback_or_invalidate())
                raise

    async def execute(self, statement, params=None, **kwargs):
        """Cancellation-safe execute.

        execute() is the workhorse: every SELECT/INSERT/UPDATE/DELETE on
        the session goes through here (including the ones that
        ``scalars()``, ``scalar()``, ``get()`` etc. call internally).

        Tear-in-the-middle hazard: a CancelledError between Parse/Bind
        and Execute/Sync leaves the asyncpg backend in state=active
        wait_event=Client/ClientRead holding row locks the statement
        acquired. The 4-min zombie sweeper eventually clears those, but
        in the meantime any other writer touching the same rows blocks.

        Mitigation has two parts:
          1. Run the underlying ``super().execute(...)`` as an explicit
             Task and shield it. CancelledError propagates to the caller
             but the inner Task keeps running until the extended-protocol
             sequence completes atomically.
          2. On cancel, schedule a fire-and-forget cleanup that *waits
             for the inner task to drain* before invalidating the
             connection. This is critical: invalidating while the inner
             task is mid-protocol triggers asyncpg
             ``InternalClientError: got result for unknown protocol
             state`` because two coroutines end up sharing the same
             connection. Drain first, then invalidate, then return —
             zero cross-task corruption.
        """
        # Drain any prior shielded inner left by a cancelled call on
        # this session; without this a wait_for-cancelled execute on the
        # same session would surface isce / "another operation in
        # progress" the next time we touch the connection.
        from utils.retry import is_db_connection_broken

        await self._wait_inflight()
        inner = asyncio.ensure_future(super().execute(statement, params=params, **kwargs))
        self._track_inflight(inner)
        try:
            return await asyncio.shield(inner)
        except asyncio.CancelledError:
            self._fire_and_forget(self._drain_then_invalidate(inner))
            raise
        except DBAPIError as exc:
            # Connection-broken errors (e.g. asyncpg
            # ``cannot switch to state X; another operation in progress``)
            # mean the underlying socket's protocol state is corrupted —
            # rollback would race the still-active inner task and the
            # next checkout would reuse the poisoned connection.  Drain
            # the inner first, then drop the connection so the pool
            # replaces it.
            if is_db_connection_broken(exc):
                self._fire_and_forget(self._drain_then_invalidate(inner))
                raise
            try:
                await self._reset_after_failed_execute()
            except Exception:
                pass
            raise
        except Exception as exc:
            # Catch raw driver-level errors that SQLAlchemy did NOT
            # wrap as ``DBAPIError`` — most importantly
            # ``asyncpg.exceptions._base.InternalClientError`` ("cannot
            # switch to state X; another operation in progress").  This
            # is asyncpg's INTERNAL error, surfaced when the protocol
            # state is corrupted by a cancellation that landed mid
            # extended-protocol exchange.  It is NOT a subclass of
            # SQLAlchemy DBAPIError, so the previous handler missed it
            # and the poisoned connection went back to the pool — every
            # subsequent checkout produced the same error.  We invalidate
            # via drain-then-invalidate so the pool drops the socket.
            if is_db_connection_broken(exc):
                self._fire_and_forget(self._drain_then_invalidate(inner))
            raise

    async def flush(self, objects=None) -> None:
        """Cancellation-safe flush.

        flush() is where session.add() rows actually get sent to the
        server as INSERT/UPDATE statements over the asyncpg extended
        protocol.  Same tear-in-the-middle hazard as execute() — see
        execute() docstring for the drain-then-invalidate rationale.
        """
        from utils.retry import is_db_connection_broken

        await self._wait_inflight()
        inner = asyncio.ensure_future(super().flush(objects))
        self._track_inflight(inner)
        try:
            await asyncio.shield(inner)
        except asyncio.CancelledError:
            self._fire_and_forget(self._drain_then_invalidate(inner))
            raise
        except Exception as exc:
            # Connection-broken errors: drain then invalidate (drop the
            # poisoned socket); other errors: same drain-then-invalidate
            # path because flush errors generally indicate the session
            # is in a state we shouldn't reuse.  See ``execute()`` and
            # ``commit()`` for the same pattern.
            _ = is_db_connection_broken(exc)  # marker check (no branch needed; same action)
            self._fire_and_forget(self._drain_then_invalidate(inner))
            raise

    # ------------------------------------------------------------------
    # Internal helpers (run inside fire-and-forget tasks)
    # ------------------------------------------------------------------

    async def _do_close_or_invalidate(self) -> None:
        """Close the session; fall back to invalidate on any error.

        Three cases of inflight cleanup, all of which must invalidate
        rather than close:

          1. **Drain timed out** — the inner task is still running after
             ``_DRAIN_TIMEOUT_SECONDS``.  ``super().close()`` would
             race the still-active protocol; invalidate drops the
             connection cleanly.

          2. **Drain succeeded with a connection-broken result** —
             this was the gap that caused the persistent 2026-04-28
             ``cannot switch to state X; another operation in
             progress`` cascade.  An inner task that completed with
             that asyncpg error has left the underlying socket in a
             corrupted protocol state.  ``super().close()`` would
             return the poisoned connection to the pool, where the
             next checkout reuses it and produces the same error.
             We must invalidate so the pool drops it.

          3. **No inflight tasks** — clean close, return to pool.
        """
        from utils.retry import is_db_connection_broken

        bag = getattr(self, "_inflight_tasks", None)
        drain_timed_out = False
        connection_poisoned = False
        if bag:
            inflight = [t for t in list(bag) if not t.done()]
            if inflight:
                try:
                    _done, pending = await asyncio.wait(
                        inflight, timeout=self._DRAIN_TIMEOUT_SECONDS
                    )
                    drain_timed_out = bool(pending)
                except Exception:
                    drain_timed_out = True
            # Inspect the (now-completed) inner tasks.  If any of them
            # raised a connection-broken error (the asyncpg protocol
            # corruption case), the underlying socket is in a state
            # where the next caller will hit the same error.  Mark the
            # connection poisoned so we invalidate instead of close.
            for task in list(bag):
                if not task.done():
                    continue
                try:
                    task_exc = task.exception()
                except (asyncio.CancelledError, asyncio.InvalidStateError):
                    task_exc = None
                if task_exc is not None and is_db_connection_broken(task_exc):
                    connection_poisoned = True
                    break
        if drain_timed_out or connection_poisoned:
            try:
                await super().invalidate()
            except Exception:
                pass
            return
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

    async def _drain_then_invalidate(self, inner) -> None:
        """Wait for a shielded inner task to finish, then invalidate.

        Invalidating while the inner protocol exchange is still in flight
        causes two coroutines to race for the asyncpg connection lock,
        which surfaces as ``InternalClientError: got result for unknown
        protocol state``. Draining the inner task first guarantees the
        connection is at a quiescent point before invalidate touches it.
        """
        try:
            await inner
        except Exception:
            pass
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

    async def _reset_after_failed_execute(self) -> None:
        try:
            if self.in_transaction():
                await self.rollback()
            else:
                await self.invalidate()
        except Exception:
            try:
                await self.invalidate()
            except Exception:
                pass


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
    """Simulated trading account for shadow trading"""

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


# ==================== ML TRAINING DATA ====================


class MLTrainingSnapshot(Base):
    """Time-series orderbook snapshots recorded from live crypto markets for ML training.

    Captured by the crypto worker every N seconds, storing price, spread, depth,
    and volume features that can be used to train directional prediction models.
    """

    __tablename__ = "ml_training_snapshots"

    id = Column(String, primary_key=True)
    task_key = Column(String, nullable=False, default="crypto_directional")
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
        Index("idx_mlt_task_asset_tf_ts", "task_key", "asset", "timeframe", "timestamp"),
        Index("idx_mlt_asset_tf_ts", "asset", "timeframe", "timestamp"),
        Index("idx_mlt_timestamp", "timestamp"),
        Index("idx_mlt_asset", "asset"),
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


class MachineLearningModelArtifact(Base):
    __tablename__ = "machine_learning_model_artifacts"

    id = Column(String, primary_key=True)
    task_key = Column(String, nullable=False)
    name = Column(String, nullable=False)
    backend = Column(String, nullable=False)
    status = Column(String, nullable=False, default="ready")
    artifact_path = Column(String, nullable=False)
    artifact_sha256 = Column(String, nullable=False)
    manifest_json = Column(JSON, nullable=False)
    metrics_json = Column(JSON, nullable=True)
    source_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=_utcnow, nullable=False)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow, nullable=False)
    archived_at = Column(DateTime, nullable=True)

    __table_args__ = (
        Index("idx_mla_task_status", "task_key", "status"),
        Index("idx_mla_created", "created_at"),
        UniqueConstraint("task_key", "name", name="uq_mla_task_name"),
    )


class MachineLearningAdapterArtifact(Base):
    __tablename__ = "machine_learning_adapter_artifacts"

    id = Column(String, primary_key=True)
    task_key = Column(String, nullable=False)
    base_model_id = Column(String, ForeignKey("machine_learning_model_artifacts.id"), nullable=False)
    name = Column(String, nullable=False)
    adapter_type = Column(String, nullable=False)
    status = Column(String, nullable=False, default="ready")
    artifact_path = Column(String, nullable=False)
    artifact_sha256 = Column(String, nullable=False)
    manifest_json = Column(JSON, nullable=False)
    metrics_json = Column(JSON, nullable=True)
    training_source_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=_utcnow, nullable=False)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow, nullable=False)
    archived_at = Column(DateTime, nullable=True)

    base_model = relationship("MachineLearningModelArtifact")

    __table_args__ = (
        Index("idx_mlad_task_status", "task_key", "status"),
        Index("idx_mlad_base_model", "base_model_id"),
        Index("idx_mlad_created", "created_at"),
        UniqueConstraint("task_key", "name", name="uq_mlad_task_name"),
    )


class MachineLearningDeployment(Base):
    __tablename__ = "machine_learning_deployments"

    id = Column(String, primary_key=True)
    task_key = Column(String, nullable=False)
    base_model_id = Column(String, ForeignKey("machine_learning_model_artifacts.id"), nullable=False)
    adapter_id = Column(String, ForeignKey("machine_learning_adapter_artifacts.id"), nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=_utcnow, nullable=False)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow, nullable=False)
    activated_at = Column(DateTime, default=_utcnow, nullable=False)

    base_model = relationship("MachineLearningModelArtifact", foreign_keys=[base_model_id])
    adapter = relationship("MachineLearningAdapterArtifact", foreign_keys=[adapter_id])

    __table_args__ = (
        Index("idx_mld_updated", "updated_at"),
        UniqueConstraint("task_key", name="uq_mld_task_key"),
    )


class MachineLearningJob(Base):
    __tablename__ = "machine_learning_jobs"

    id = Column(String, primary_key=True)
    task_key = Column(String, nullable=False)
    job_type = Column(String, nullable=False)
    status = Column(String, nullable=False, default="queued")
    target_id = Column(String, nullable=True)
    message = Column(String, nullable=True)
    error = Column(Text, nullable=True)
    payload_json = Column(JSON, nullable=True)
    result_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=_utcnow, nullable=False)
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)

    __table_args__ = (
        Index("idx_mlj_task_created", "task_key", "created_at"),
        Index("idx_mlj_status", "status"),
        Index("idx_mlj_created", "created_at"),
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


class MarketTagSeen(Base):
    """Distinct tags observed on Polymarket / Kalshi markets and events.

    Populated by ``services.market_tag_aggregator.record_tags_from_markets``
    on every ingest cycle, before any filter is applied. Feeds the
    operator-facing tag chooser in ``Settings → Scanner → Market Tag
    Filter``. Old rows are purged by ``prune_stale_tags`` once a day.
    """

    __tablename__ = "market_tags_seen"

    tag = Column(String, primary_key=True)
    first_seen = Column(DateTime, nullable=False, default=_utcnow)
    last_seen = Column(DateTime, nullable=False, default=_utcnow, onupdate=_utcnow)
    occurrences = Column(BigInteger, nullable=False, default=1)

    __table_args__ = (
        Index("idx_market_tags_seen_last_seen", "last_seen"),
    )


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

    # Chainlink Data Streams (direct REST polling). Optional second source
    # alongside the RTDS-relayed Chainlink prices — when both are present,
    # the source-comparison panel surfaces the relay-vs-direct delta.
    chainlink_direct_api_key = Column(String, nullable=True)
    chainlink_direct_user_secret = Column(String, nullable=True)

    # LLM/AI Service Settings
    openai_api_key = Column(String, nullable=True)
    anthropic_api_key = Column(String, nullable=True)
    llm_provider = Column(String, default="none")  # none, openai, anthropic, google, xai, deepseek, openrouter, ollama, lmstudio
    llm_model = Column(String, nullable=True)
    google_api_key = Column(String, nullable=True)
    xai_api_key = Column(String, nullable=True)
    deepseek_api_key = Column(String, nullable=True)
    openrouter_api_key = Column(String, nullable=True)
    openrouter_base_url = Column(String, nullable=True)
    ollama_api_key = Column(String, nullable=True)
    ollama_base_url = Column(String, nullable=True)
    lmstudio_api_key = Column(String, nullable=True)
    lmstudio_base_url = Column(String, nullable=True)
    nvidia_api_key = Column(String, nullable=True)
    nvidia_base_url = Column(String, nullable=True)

    # AI Feature Settings
    ai_enabled = Column(Boolean, default=False)  # Master switch for AI features
    ai_resolution_analysis = Column(Boolean, default=True)  # Auto-analyze resolution criteria
    ai_opportunity_scoring = Column(Boolean, default=True)  # LLM-as-judge scoring
    ai_news_sentiment = Column(Boolean, default=True)  # News/sentiment analysis
    ai_max_monthly_spend = Column(Float, default=50.0)  # Monthly LLM cost cap
    ai_default_model = Column(String, default="gpt-4o-mini")  # Default model for AI tasks
    ai_premium_model = Column(String, default="gpt-4o")  # Model for high-value analysis
    llm_model_assignments = Column(JSON, nullable=True)  # Per-purpose model overrides
    llm_enabled_features = Column(JSON, nullable=True)  # Per-feature LLM enable/disable

    # External market data providers (Data Lab on-demand import).  All
    # nullable; the UI surfaces these in Settings → Data Sources →
    # Providers.  No defaults baked in code so the operator's choice in
    # the UI is the single source of truth.
    polybacktest_api_key = Column(String, nullable=True)
    polybacktest_base_url = Column(String, nullable=True)

    # Strategy reverse-engineer agent — UI-tunable defaults.  Null
    # values fall back to service-level constants.  The default *model*
    # for this purpose lives in ``llm_model_assignments['strategy_reverse_engineer']``
    # — same JSON column the AI tab → Models view manages for every other
    # per-purpose LLM override (chat, news_analysis, etc.).
    reverse_engineer_max_iterations = Column(Integer, nullable=True)
    reverse_engineer_target_score = Column(Float, nullable=True)
    reverse_engineer_max_cost_usd = Column(Float, nullable=True)
    reverse_engineer_max_wallet_trades = Column(Integer, nullable=True)

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
    scanner_skipped_signal_reactivation_cooldown_seconds = Column(Integer, default=180)
    scanner_strict_ws_max_age_ms = Column(Integer, default=30000)

    # Market Tag Filter (whitelist applied at Polymarket/Kalshi ingest before
    # the catalog is written). Empty / null list = filter inactive (no markets
    # dropped on tag). Stored already-normalised: lowercased, trimmed, deduped.
    # ``market_filter_updated_at`` is the audit timestamp of the last operator
    # write — surfaced in the API payload so the UI can render "last changed".
    market_filter_tags = Column(JSON, nullable=True)
    market_filter_updated_at = Column(DateTime, nullable=True)

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

    # CTF Redeemer Policy (UI-tunable; see alembic 202604280001)
    redeemer_min_payout_usd = Column(Float, nullable=True)
    redeemer_max_gas_price_gwei = Column(Float, nullable=True)
    redeemer_force_including_losers = Column(Boolean, nullable=True)

    # Polymarket collateral selection — default for operator-initiated
    # split/merge calls. Recognized values: ``pusd``, ``usdc.e``,
    # ``usdc_native``. Redemption auto-detects collateral per-position
    # from chain math regardless of this value (see alembic 202604300004).
    polymarket_default_collateral = Column(String, nullable=True)

    # Operator-tunable latency fallbacks for the fill simulator + the
    # BacktestStudio "Latency (defaults)" panel.  When no real submit/
    # cancel latencies have been measured in the last 15 min, these
    # values are used as the conservative envelope.  NULL columns
    # fall back to the module-level constants (200/600/1500 ms).
    latency_fallback_p50_ms = Column(Float, nullable=True)
    latency_fallback_p95_ms = Column(Float, nullable=True)
    latency_fallback_p99_ms = Column(Float, nullable=True)

    # Search Settings (which platforms to query, result limits)
    search_polymarket_enabled = Column(Boolean, default=True)
    search_kalshi_enabled = Column(Boolean, default=False)
    search_max_results = Column(Integer, default=50)

    # Web Search Provider API Keys
    serpapi_key = Column(String, nullable=True)
    brave_search_key = Column(String, nullable=True)

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

    # Liquidity Vacuum
    liquidity_vacuum_enabled = Column(Boolean, default=True)
    liquidity_vacuum_min_imbalance_ratio = Column(Float, default=5.0)
    liquidity_vacuum_min_depth_usd = Column(Float, default=100.0)

    # Entropy Arbitrage
    entropy_arb_enabled = Column(Boolean, default=True)
    entropy_arb_min_deviation = Column(Float, default=0.25)

    # Event-Driven Arbitrage
    event_driven_enabled = Column(Boolean, default=True)

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
    cleanup_trade_signal_days = Column(Integer, default=30)
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

    # Polygon RPC endpoints used by services.wallet_ws_monitor for
    # eth_getLogs polling (when WS connection is unavailable) and the
    # canonical on-chain wallet trade detection.  Stored encrypted
    # because Ankr/Alchemy/QuickNode embed the API key in the URL path
    # (https://rpc.ankr.com/polygon/<32-hex-key>).  When unset, falls
    # back to the POLYGON_RPC_URL / POLYGON_WS_URL env vars and finally
    # to the hardcoded public-tier list in wallet_ws_monitor.py.
    polygon_rpc_url = Column(String, nullable=True)  # encrypted via set_encrypted_secret
    polygon_ws_url = Column(String, nullable=True)   # encrypted via set_encrypted_secret

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

    # Cortex Agent (autonomous fleet commander)
    cortex_enabled = Column(Boolean, default=False)
    cortex_model = Column(String, nullable=True)  # LLM model override; None = use ai_default_model
    cortex_interval_seconds = Column(Integer, default=300)  # run every N seconds
    cortex_max_iterations = Column(Integer, default=15)
    cortex_temperature = Column(Float, default=0.1)
    cortex_mandate = Column(Text, nullable=True)  # custom system prompt override
    cortex_memory_limit = Column(Integer, default=20)  # max memories injected per run
    cortex_write_actions_enabled = Column(Boolean, default=False)  # allow strategy/trader mutations
    cortex_notify_telegram = Column(Boolean, default=False)  # send Telegram on actions

    # Autoresearch loop (Karpathy-inspired continuous param optimization)
    autoresearch_model = Column(String, nullable=True)  # LLM model override; None = use ai_default_model
    autoresearch_max_iterations = Column(Integer, default=50)  # max iterations per experiment
    autoresearch_interval_seconds = Column(Integer, default=600)  # auto-trigger interval
    autoresearch_temperature = Column(Float, default=0.2)
    autoresearch_mandate = Column(Text, nullable=True)  # custom constraints/instructions
    autoresearch_auto_apply = Column(Boolean, default=True)  # auto-apply kept params to trader
    autoresearch_walk_forward_windows = Column(Integer, default=5)
    autoresearch_train_ratio = Column(Float, default=0.7)
    autoresearch_mode = Column(String, default="params")  # "params" or "code"

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


class StrategyPersistentState(Base):
    """Durable per-strategy key/value state.

    Backs ``StrategySDK.PersistentState`` — gives custom strategies a
    place to persist data across worker restarts (rolling stats, last
    seen timestamps, multi-window state, etc.) without each strategy
    inventing its own table. ``self.state`` on BaseStrategy is in-memory
    only; this is the durable counterpart.

    Composite PK on ``(strategy_slug, key)``. Values are JSON, so any
    JSON-serialisable Python value can be stored. Strategies access
    rows through the SDK helper, never directly.
    """

    __tablename__ = "strategy_persistent_state"

    strategy_slug = Column(String, primary_key=True)
    key = Column(String, primary_key=True)
    value = Column(JSON, nullable=False, default=dict)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow, nullable=False)

    __table_args__ = (
        Index("idx_strategy_persistent_state_slug", "strategy_slug"),
        Index("idx_strategy_persistent_state_updated", "updated_at"),
    )


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
    provider = Column(String, nullable=False)  # openai, anthropic, google, xai, deepseek, openrouter, ollama, lmstudio
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


class UserAgent(Base):
    """User-defined AI agent configuration."""

    __tablename__ = "user_agents"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(String, nullable=True)
    system_prompt = Column(Text, nullable=False)
    tools = Column(JSON, nullable=False, default=list)
    model = Column(String, nullable=True)
    temperature = Column(Float, nullable=False, default=0.0)
    max_iterations = Column(Integer, nullable=False, default=10)
    is_builtin = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)


class UserTool(Base):
    """User-defined or builtin tool that can be assigned to AI agents."""

    __tablename__ = "user_tools"

    id = Column(String, primary_key=True, default=lambda: str(__import__("uuid").uuid4()))
    name = Column(String, nullable=False, unique=True)
    description = Column(Text, nullable=True)
    tool_type = Column(String, default="function")  # function, api, etc
    parameters_schema = Column(JSON, nullable=True)  # JSON schema for params
    implementation = Column(Text, nullable=True)  # Python code or API config
    is_builtin = Column(Boolean, default=False)
    enabled = Column(Boolean, default=True)
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)


class LLMUsageLog(Base):
    """Tracks LLM API usage for cost management and observability.

    Every LLM API call is logged here with token counts, costs,
    latency, and error information. Used for spend tracking,
    rate limiting, and debugging.
    """

    __tablename__ = "llm_usage_log"

    id = Column(String, primary_key=True)
    provider = Column(String, nullable=False)  # openai, anthropic, google, xai, deepseek, openrouter, ollama, lmstudio
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


# ==================== CORTEX AGENT ====================


class CortexMemory(Base):
    """Persistent memory entry for the Cortex fleet commander agent.

    Stores observations, lessons, rules, and preferences that the agent
    accumulates over time to improve its fleet management decisions.
    """

    __tablename__ = "cortex_memory"

    id = Column(String, primary_key=True, default=lambda: str(__import__("uuid").uuid4()))
    category = Column(String, nullable=False, default="observation")  # observation, lesson, rule, preference
    content = Column(Text, nullable=False)
    context_json = Column(JSON, nullable=True)  # structured metadata at time of writing
    importance = Column(Float, nullable=False, default=0.5)  # 0-1 relevance score
    access_count = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)
    expired = Column(Boolean, nullable=False, default=False)

    __table_args__ = (
        Index("idx_cortex_memory_category", "category"),
        Index("idx_cortex_memory_importance", "importance"),
        Index("idx_cortex_memory_expired", "expired"),
    )


class CortexRunLog(Base):
    """Full audit trail for every autonomous Cortex agent run.

    Captures the complete reasoning trace, tool calls, actions taken,
    learnings saved, and cost for each cycle.
    """

    __tablename__ = "cortex_run_log"

    id = Column(String, primary_key=True, default=lambda: str(__import__("uuid").uuid4()))
    started_at = Column(DateTime, default=_utcnow)
    finished_at = Column(DateTime, nullable=True)
    status = Column(String, nullable=False, default="running")  # running, completed, error, timeout
    thinking_log = Column(Text, nullable=True)  # full reasoning trace
    actions_taken = Column(JSON, nullable=True)  # [{tool, input, output}, ...]
    learnings_saved = Column(JSON, nullable=True)  # [{id, category, content}, ...]
    summary = Column(Text, nullable=True)  # agent's final summary
    tokens_used = Column(Integer, nullable=True, default=0)
    cost_usd = Column(Float, nullable=True, default=0.0)
    model_used = Column(String, nullable=True)
    trigger = Column(String, nullable=False, default="scheduled")  # scheduled, manual

    __table_args__ = (
        Index("idx_cortex_run_started", "started_at"),
        Index("idx_cortex_run_status", "status"),
    )


# ==================== AUTORESEARCH ====================


class AutoresearchExperiment(Base):
    """An autoresearch optimization experiment.

    Two scopes:
      * **trader-scoped** (``trader_id`` set): per-bot parameter tuning
        that operates on a specific bot's live ``strategy_params``. The
        experiment proposes tweaks to the bot's running config.
      * **strategy-scoped** (``strategy_id`` set, ``trader_id`` NULL):
        per-strategy code evolution that operates on the backtest data
        plane only. No bot is involved — the experiment evolves the
        strategy's source code and kept versions are bumped on the
        Strategy record itself.

    Either ``trader_id`` or ``strategy_id`` must be set; both may be
    set when a code experiment is initiated from a bot context.
    """

    __tablename__ = "autoresearch_experiments"

    id = Column(String, primary_key=True, default=lambda: str(__import__("uuid").uuid4()))
    trader_id = Column(
        String, ForeignKey("traders.id", ondelete="CASCADE"), nullable=True
    )
    name = Column(String, nullable=False)
    status = Column(String, nullable=False, default="running")  # running, paused, completed, failed
    mode = Column(String, default="params")  # "params" | "code"
    strategy_id = Column(String, ForeignKey("strategies.id"), nullable=True)  # target strategy for code mode
    baseline_score = Column(Float, default=0.0)
    best_score = Column(Float, default=0.0)
    best_params_json = Column(JSON, nullable=True)
    best_source_code = Column(Text, nullable=True)  # best evolved source (code mode)
    best_version = Column(Integer, nullable=True)  # StrategyVersion number of best code
    iteration_count = Column(Integer, default=0)
    kept_count = Column(Integer, default=0)
    reverted_count = Column(Integer, default=0)
    settings_json = Column(JSON, nullable=True)  # snapshot of model, mandate, walk-forward config
    started_at = Column(DateTime, default=_utcnow)
    finished_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=_utcnow, nullable=False)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)

    __table_args__ = (
        Index("idx_arx_trader_id", "trader_id"),
        Index("idx_arx_strategy_id", "strategy_id"),
        Index("idx_arx_status", "status"),
    )


class AutoresearchIteration(Base):
    """A single iteration within an autoresearch experiment.

    Records the proposed param changes, backtest result, score comparison,
    and whether the change was kept or reverted.
    """

    __tablename__ = "autoresearch_iterations"

    id = Column(String, primary_key=True, default=lambda: str(__import__("uuid").uuid4()))
    experiment_id = Column(String, ForeignKey("autoresearch_experiments.id"), nullable=False)
    iteration_number = Column(Integer, nullable=False)
    proposed_params_json = Column(JSON, nullable=True)
    baseline_score = Column(Float, default=0.0)
    new_score = Column(Float, default=0.0)
    score_delta = Column(Float, default=0.0)
    decision = Column(String, nullable=False)  # "kept" | "reverted"
    reasoning = Column(Text, nullable=True)
    backtest_result_json = Column(JSON, nullable=True)
    changed_params_json = Column(JSON, nullable=True)  # only the params that changed
    source_code_snapshot = Column(Text, nullable=True)  # full proposed source (code mode)
    source_diff = Column(Text, nullable=True)  # unified diff vs baseline (code mode)
    validation_result_json = Column(JSON, nullable=True)  # AST validation output (code mode)
    duration_seconds = Column(Float, default=0.0)
    tokens_used = Column(Integer, default=0)
    created_at = Column(DateTime, default=_utcnow, nullable=False)

    __table_args__ = (
        Index("idx_ari_experiment_id", "experiment_id"),
        Index("idx_ari_experiment_iteration", "experiment_id", "iteration_number"),
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
        Index(
            "idx_discovered_pool_rank_order",
            text("composite_score DESC NULLS LAST"),
            text("rank_score DESC NULLS LAST"),
            text("total_pnl DESC NULLS LAST"),
            text("address ASC"),
        ),
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
        Index("idx_trader_group_member_wallet_lower", text("lower(wallet_address)")),
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
    pending_reconciliation_json = Column(JSON, default=list)
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
    market_id = Column(String, nullable=True, index=True)
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
    redeemable = Column(Boolean, nullable=False, default=False, server_default="false")
    counts_as_open = Column(Boolean, nullable=False, default=True, server_default="true")
    end_date = Column(String, nullable=True)
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
    raw_detected_count = Column(Integer, default=0)
    displayable_count = Column(Integer, default=0)
    execution_eligible_count = Column(Integer, default=0)
    opportunities_count = Column(Integer, default=0)
    # Status fields (denormalized for API)
    running = Column(Boolean, default=True)
    enabled = Column(Boolean, default=True)
    current_activity = Column(String, nullable=True)
    interval_seconds = Column(Integer, default=60)
    strategies_json = Column(JSON, default=list)  # list of {name, type}
    strategy_diagnostics_json = Column(JSON, default=dict)
    tiered_scanning_json = Column(JSON, nullable=True)
    ws_feeds_json = Column(JSON, nullable=True)


class ScannerMarketHistory(Base):
    __tablename__ = "scanner_market_history"

    market_id = Column(String, primary_key=True)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)
    points_json = Column(JSON, default=list)


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
    runtime_sequence = Column(BigInteger, nullable=True, index=True)
    created_at = Column(DateTime, default=_utcnow, nullable=False)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)

    __table_args__ = (
        Index("idx_trade_signals_created", "created_at"),
        # 2026-05-09: dropped idx_trade_signals_source_status — strict
        # prefix of idx_trade_signals_source_status_sequence, postgres
        # uses the longer index for any query the shorter one served.
        # Eliminates one B-tree write per non-HOT UPDATE.
        #
        # The active-status index is PARTIAL (status IN active states
        # only). Terminal rows (executed/skipped/expired/failed) are
        # ~99% of the table but never queried by hot-path code.
        #
        # 2026-05-09 (#2): dropped runtime_sequence from the indexed
        # column set so producer UPSERTs (which always set
        # runtime_sequence) become HOT-eligible. ORDER BY
        # runtime_sequence in list_unconsumed_trade_signals becomes a
        # Sort node above the (source, status) scan -- sub-millisecond
        # over the ~3,800 active rows. Renamed the index from
        # ``..._sequence`` to ``..._active`` so the name reflects the
        # actual columns.
        Index(
            "idx_trade_signals_source_status_active",
            "source", "status",
            postgresql_where=text(
                "status IN ('pending', 'selected', 'submitted')"
            ),
        ),
        Index(
            "idx_trade_signals_market_status",
            "market_id", "status",
            postgresql_where=text(
                "status IN ('pending', 'selected', 'submitted')"
            ),
        ),
        UniqueConstraint("source", "dedupe_key", name="uq_trade_signals_source_dedupe"),
    )


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


class MarketMicrostructureSnapshot(Base):
    """Durable top-of-book, depth, and trade tape samples for execution replay."""

    __tablename__ = "market_microstructure_snapshots"

    id = Column(String, primary_key=True)
    provider = Column(String, nullable=False, default="polymarket", index=True)
    token_id = Column(String, nullable=False, index=True)
    snapshot_type = Column(String, nullable=False, index=True)
    observed_at = Column(DateTime, nullable=False, index=True)
    exchange_ts_ms = Column(BigInteger, nullable=True)
    sequence = Column(BigInteger, nullable=True)
    best_bid = Column(Float, nullable=True)
    best_ask = Column(Float, nullable=True)
    spread_bps = Column(Float, nullable=True)
    bids_json = Column(JSON, nullable=True)
    asks_json = Column(JSON, nullable=True)
    trade_price = Column(Float, nullable=True)
    trade_size = Column(Float, nullable=True)
    trade_side = Column(String, nullable=True)
    payload_json = Column(JSON, default=dict)
    created_at = Column(DateTime, default=_utcnow, nullable=False)

    __table_args__ = (
        Index("idx_mms_token_observed", "token_id", "observed_at"),
        Index("idx_mms_token_type_observed", "token_id", "snapshot_type", "observed_at"),
    )


class BookDeltaEvent(Base):
    """Tick-by-tick decomposition of order book delta events.

    Every WebSocket update is classified into ``trade`` (a print at the
    same price level cleared real depth) or ``cancel`` (depth disappeared
    with no matching trade — interpreted as a cancellation).  This lets
    the fill simulator distinguish queue-advancing fills from queue-
    advancing cancels, which have very different implications for adverse
    selection.

    Trades carry a non-null ``trade_price`` and ``trade_size``; cancels
    carry the level + side that emptied with ``cancel_size`` set.

    Persisted by the WebSocket consumer in services/ws_feeds.py — see
    ``LiveMarketDataIngestor`` in services/market_data_ingestor.py
    (formerly the standalone ``BookDeltaDecomposer``).
    """

    __tablename__ = "book_delta_events"

    id = Column(String, primary_key=True)
    provider = Column(String, nullable=False, default="polymarket", index=True)
    token_id = Column(String, nullable=False, index=True)
    observed_at = Column(DateTime, nullable=False, index=True)
    exchange_ts_ms = Column(BigInteger, nullable=True)
    sequence = Column(BigInteger, nullable=True)
    event_type = Column(String, nullable=False, index=True)  # "trade" | "cancel"
    side = Column(String, nullable=True)  # "bid" | "ask"
    price = Column(Float, nullable=False)
    trade_size = Column(Float, nullable=True)  # filled size at this print
    cancel_size = Column(Float, nullable=True)  # depth that disappeared without trade
    queue_depth_before = Column(Float, nullable=True)  # depth at price BEFORE event
    queue_depth_after = Column(Float, nullable=True)
    spread_bps_at_event = Column(Float, nullable=True)
    payload_json = Column(JSON, default=dict)
    created_at = Column(DateTime, default=_utcnow, nullable=False)

    __table_args__ = (
        Index("idx_bde_token_observed", "token_id", "observed_at"),
        Index("idx_bde_token_type_observed", "token_id", "event_type", "observed_at"),
    )


class FillProbabilityModel(Base):
    """Versioned Cox proportional hazards (or Kaplan-Meier fallback) fill model.

    Trained nightly by ``workers/cox_trainer_worker.py`` from
    ``trader_orders`` joined against ``market_microstructure_snapshots``
    at placement time.  Each row is one promoted model; the ``active``
    flag picks which one inference reads.
    """

    __tablename__ = "fill_probability_models"

    id = Column(String, primary_key=True)
    family = Column(String, nullable=False, default="cox_ph", index=True)  # "cox_ph" | "kaplan_meier"
    strata_key = Column(String, nullable=False, default="pooled", index=True)  # e.g. "pooled" | "crypto_15m" | "crypto_60m"
    trained_at = Column(DateTime, default=_utcnow, nullable=False, index=True)
    training_window_start = Column(DateTime, nullable=True)
    training_window_end = Column(DateTime, nullable=True)
    n_events = Column(Integer, nullable=False, default=0)
    n_observations = Column(Integer, nullable=False, default=0)
    concordance_index = Column(Float, nullable=True)
    brier_score = Column(Float, nullable=True)
    log_likelihood = Column(Float, nullable=True)
    coefficients_json = Column(JSON, nullable=False, default=dict)  # {covariate: hazard_ratio}
    baseline_survival_json = Column(JSON, nullable=True)  # KM-style baseline S(t)
    feature_means_json = Column(JSON, nullable=True)  # for centering at inference
    feature_stds_json = Column(JSON, nullable=True)  # for standardizing
    config_json = Column(JSON, default=dict)  # hyperparams, covariate list
    promoted_at = Column(DateTime, nullable=True)
    active = Column(Boolean, default=False, nullable=False, index=True)
    notes = Column(String, nullable=True)

    __table_args__ = (
        Index("idx_fpm_family_strata_active", "family", "strata_key", "active"),
        Index("idx_fpm_active_trained", "active", "trained_at"),
    )


class BacktestRun(Base):
    """Persisted Backtest Studio run + job-queue row.

    Holds the full run lifecycle: from initial enqueue (operator clicks
    Run) through worker claim, in-flight progress, and final result.
    The split from "run" to "run + job" is deliberate — having a single
    canonical row eliminates the need for a separate jobs table and a
    join to read run state.

    Lifecycle (``status`` column):

      queued    → enqueued, waiting for a worker to claim it
      running   → claimed by a worker; ``progress`` 0.0→1.0,
                  ``message`` carries human-readable activity
      completed → engine finished; ``result_json`` populated
      failed    → crashed; ``error`` populated
      cancelled → operator clicked stop; worker bails on next yield

    The dedicated backtest worker process (workers/backtest_worker.py)
    polls this table for ``status='queued'`` rows and runs them off
    the API event loop entirely — guarantees the orchestrator and
    other workers cannot be impacted by a long-running backtest.

    ``payload_json`` carries the full run request shape:
        { source_code, slug, config, token_ids, start, end,
          initial_capital_usd, ... } so the worker can reconstruct
        the run without reading other tables.

    Legacy in-process runs (POST /backtest/run sync path) write rows
    with ``status='ok'`` directly, skipping the queue lifecycle.  The
    UI tolerates both code paths.
    """

    __tablename__ = "backtest_runs"

    id = Column(String, primary_key=True)
    strategy_slug = Column(String, nullable=True)
    strategy_name = Column(String, nullable=True)
    started_at = Column(DateTime, nullable=False, index=True)
    completed_at = Column(DateTime, nullable=True)
    total_time_ms = Column(Float, nullable=False, default=0.0)
    # Status values: queued | running | completed | failed | cancelled |
    # ok (legacy sync path).  ``ok`` is treated as ``completed`` by
    # readers; we keep both so old rows still work.
    status = Column(String, nullable=False, default="ok", index=True)
    trade_count = Column(Integer, nullable=False, default=0)
    total_return_pct = Column(Float, nullable=False, default=0.0)
    sparkline_pct_json = Column(JSON, nullable=True)  # list[float]
    result_json = Column(JSON, nullable=False, default=dict)

    # ── Job-queue lifecycle fields ─────────────────────────────────
    # ``payload_json`` is set at enqueue time so the worker can rebuild
    # the run config without dependencies on the API process state.
    payload_json = Column(JSON, nullable=True)
    # 0.0 → 1.0; the engine's progress_callback writes here every
    # ~1k snapshots so the UI can render a live progress bar.
    progress = Column(Float, nullable=False, default=0.0)
    # Human-readable activity string ("Running engine: 47% · 8 fills").
    message = Column(String, nullable=True)
    # Worker process that claimed this row.  Diagnostic only.
    worker_id = Column(String, nullable=True)
    # When the worker picked it up.  Null for queued / legacy rows.
    claimed_at = Column(DateTime, nullable=True)
    # Stop signal: operator → backend writes True; worker checks on
    # every progress yield and bails out cleanly.
    cancel_requested = Column(Boolean, nullable=False, default=False)
    # Failure surface.
    error = Column(Text, nullable=True)
    # Snapshot tracking from the engine.  Useful for ETA estimation.
    snapshots_processed = Column(Integer, nullable=False, default=0)
    snapshots_total_estimate = Column(Integer, nullable=True)

    created_at = Column(DateTime, default=_utcnow, nullable=False)

    __table_args__ = (
        Index("idx_btr_strategy_started", "strategy_slug", "started_at"),
        Index("idx_btr_started", "started_at"),
        # Job-queue claim path: workers SELECT WHERE status='queued'
        # ORDER BY created_at LIMIT 1 FOR UPDATE SKIP LOCKED.  This
        # composite index keeps that hot.
        Index("idx_btr_status_created", "status", "created_at"),
    )


class RecordingSession(Base):
    """On-demand market-data capture session.

    Operator-triggered (or scheduled) capture of book + trade + delta
    data for a specific set of markets and a specific time window.
    Surfaced in Research → Data Lab → Record mode and consumable by
    the unified backtester via ``session_id``: the backtester scopes
    its replay to ``target_token_ids`` × ``[started_at, ended_at]``.

    The session row is metadata only — captured rows continue to
    live in ``MarketMicrostructureSnapshot`` / ``BookDeltaEvent``
    with no schema change there, pinned to the session implicitly
    by the (token, time-window) pair.
    """

    __tablename__ = "recording_sessions"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    # status: pending | scheduled | running | paused | completed | failed | cancelled
    status = Column(String, nullable=False, default="pending", index=True)

    # Targeting
    platform = Column(String, nullable=False, default="polymarket")
    target_kind = Column(String, nullable=False, default="token")  # token|condition|event
    target_values_json = Column(JSON, nullable=False, default=list)  # operator input
    target_token_ids_json = Column(JSON, nullable=True)  # resolved tokens

    # What to capture: subset of {"book", "trade", "delta"}
    capture_types_json = Column(JSON, nullable=False, default=list)
    tick_interval_ms = Column(Integer, nullable=False, default=500)
    retention_days = Column(Integer, nullable=True)

    # Scheduling
    scheduled_start_at = Column(DateTime, nullable=True)
    scheduled_end_at = Column(DateTime, nullable=True)
    max_duration_seconds = Column(Integer, nullable=True)
    started_at = Column(DateTime, nullable=True, index=True)
    ended_at = Column(DateTime, nullable=True)

    # Progress
    rows_captured = Column(Integer, nullable=False, default=0)
    last_capture_at = Column(DateTime, nullable=True)
    error = Column(Text, nullable=True)

    # Per-platform / per-recorder extras (book depth limit, etc.)
    config_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, nullable=False, default=_utcnow)
    updated_at = Column(DateTime, nullable=False, default=_utcnow, onupdate=_utcnow)


# ==================== EXTERNAL DATA PROVIDER CATALOG ====================


class ProviderDataset(Base):
    """Catalog entry for a dataset imported from an external data provider.

    A "provider" is any third-party data vendor we pull historical market
    data from on demand — currently ``polybacktest`` (Polymarket Up/Down
    book history + Binance reference prices) but the table is shaped to
    accept additional providers (Kaiko, Tardis, Crypto Compare, etc.)
    without schema changes.

    The actual snapshot rows continue to live in
    ``MarketMicrostructureSnapshot`` with ``provider`` set to the
    provider key — this table is the human-friendly catalog index that
    powers Data Lab's "Imported datasets" view and the Backtest Studio
    dataset picker.

    Synthetic ``token_id`` shape for non-Polymarket providers:
        ``{provider}:{coin}:{market_id}:{outcome}``
    e.g. ``polybacktest:btc:up-down-2026-05-04T15-00:up``.

    Unique on ``(provider, external_id)`` so re-importing the same
    provider+market produces an upsert, not a duplicate.
    """

    __tablename__ = "provider_datasets"

    id = Column(String, primary_key=True)
    provider = Column(String, nullable=False, index=True)  # 'polybacktest' etc
    coin = Column(String, nullable=True, index=True)  # 'btc' | 'eth' | 'sol' | None for non-crypto
    external_id = Column(String, nullable=False)  # provider's market id / slug
    external_slug = Column(String, nullable=True)
    title = Column(String, nullable=True)
    asset_class = Column(String, nullable=False, default="prediction")  # prediction | spot | futures
    # Synthetic token_ids written to MarketMicrostructureSnapshot for this dataset.
    token_ids_json = Column(JSON, nullable=False, default=list)
    start_ts = Column(DateTime, nullable=True, index=True)
    end_ts = Column(DateTime, nullable=True, index=True)
    snapshot_count = Column(Integer, nullable=False, default=0)
    trade_count = Column(Integer, nullable=False, default=0)
    last_imported_at = Column(DateTime, nullable=True)
    last_import_job_id = Column(String, nullable=True)
    payload_json = Column(JSON, default=dict)  # cached provider market metadata
    created_at = Column(DateTime, default=_utcnow, nullable=False)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow, nullable=False)

    __table_args__ = (
        UniqueConstraint("provider", "external_id", name="uq_provider_dataset_provider_extid"),
        Index("idx_provider_dataset_provider_coin", "provider", "coin"),
        Index("idx_provider_dataset_updated", "updated_at"),
    )


class ProviderImportJob(Base):
    """Async job: pull a window of data from an external provider.

    Mirrors the ``ValidationJob`` shape (status / progress / payload /
    result / error) so the existing job-tracking UI patterns transfer
    cleanly.  Worked off by ``workers/provider_import_worker.py`` on
    the discovery plane.

    payload_json shape (polybacktest):
        {
            "provider": "polybacktest",
            "coin": "btc",
            "market_ids": ["..."],
            "start_ms": 1714780800000,
            "end_ms":   1714867200000,
            "include_trades": true
        }
    """

    __tablename__ = "provider_import_jobs"

    id = Column(String, primary_key=True)
    provider = Column(String, nullable=False, index=True)
    status = Column(String, nullable=False, default="queued", index=True)
    # queued | running | completed | failed | cancelled
    progress = Column(Float, nullable=False, default=0.0)  # 0.0 — 1.0
    message = Column(String, nullable=True)
    payload_json = Column(JSON, nullable=False, default=dict)
    result_json = Column(JSON, nullable=True)
    error = Column(Text, nullable=True)
    snapshots_fetched = Column(Integer, nullable=False, default=0)
    snapshots_inserted = Column(Integer, nullable=False, default=0)
    trades_fetched = Column(Integer, nullable=False, default=0)
    api_calls = Column(Integer, nullable=False, default=0)
    bytes_downloaded = Column(BigInteger, nullable=False, default=0)
    created_at = Column(DateTime, default=_utcnow, nullable=False)
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)

    __table_args__ = (
        Index("idx_provider_import_status", "status"),
        Index("idx_provider_import_created", "created_at"),
    )


# ==================== STRATEGY REVERSE ENGINEER ====================


class StrategyReverseEngineerJob(Base):
    """Long-running job: reverse-engineer a wallet's trading strategy.

    Owned by the Strategy / Research domain (the deliverable is a
    strategy, not a wallet profile).  An LLM agent loop ingests the
    wallet's full trade history, picks an appropriate dataset (live
    recordings, recording sessions, polybacktest imports, etc.),
    iteratively writes Python source conforming to ``BaseStrategy``,
    backtests each candidate, scores it against the wallet's actual
    fills, and refines until ``target_score`` or ``max_iterations``.

    All per-iteration audit lives in ``StrategyReverseEngineerIteration``;
    this row only carries the headline status + best result.
    """

    __tablename__ = "strategy_reverse_engineer_jobs"

    id = Column(String, primary_key=True)
    wallet_address = Column(String, nullable=False, index=True)
    label = Column(String, nullable=True)  # human-friendly name for the run

    # report_mode chooses what the agent produces:
    #   'report'        — deterministic analytical report (the polyresearchrobotics-shape
    #                     deliverable: tables + LLM section narratives + PDF)
    #   'strategy_seed' — LLM agent loop that synthesizes a candidate
    #                     ``BaseStrategy`` Python class via iterative backtest scoring
    # Default is 'report' since that's the higher-value deliverable for
    # most operators; the strategy-seed mode is the legacy / advanced path.
    report_mode = Column(String, nullable=False, default="report")

    # Dataset scope chosen by the user (Data Lab integration).
    # One of: 'auto' | 'recording_session' | 'provider_dataset' | 'live'
    data_source_kind = Column(String, nullable=False, default="auto")
    # When data_source_kind != 'auto', the concrete IDs:
    recording_session_ids_json = Column(JSON, nullable=True)
    provider_dataset_ids_json = Column(JSON, nullable=True)

    # Configuration knobs (UI-tunable, no defaults hidden in code).
    llm_model = Column(String, nullable=True)  # null → use ai_default_model
    max_iterations = Column(Integer, nullable=False, default=10)
    target_score = Column(Float, nullable=False, default=0.7)
    max_cost_usd = Column(Float, nullable=True)  # null → no per-job ceiling
    max_wallet_trades = Column(Integer, nullable=False, default=2000)

    # Status: queued | profiling | importing_data | running | completed | failed | cancelled
    status = Column(String, nullable=False, default="queued", index=True)
    progress = Column(Float, nullable=False, default=0.0)
    current_iteration = Column(Integer, nullable=False, default=0)
    activity = Column(String, nullable=True)  # human-readable current step
    error = Column(Text, nullable=True)

    # Wallet profile (computed in step 1) — surfaced to UI without a re-fetch.
    wallet_profile_json = Column(JSON, nullable=True)
    wallet_trade_count = Column(Integer, nullable=False, default=0)
    wallet_window_start = Column(DateTime, nullable=True)
    wallet_window_end = Column(DateTime, nullable=True)

    # Best result (refreshed every iteration that improves the score).
    best_iteration_id = Column(String, nullable=True)
    best_score = Column(Float, nullable=True)
    best_strategy_code = Column(Text, nullable=True)
    best_strategy_class = Column(String, nullable=True)
    best_backtest_run_id = Column(String, nullable=True)

    # Cost / observability.
    total_input_tokens = Column(Integer, nullable=False, default=0)
    total_output_tokens = Column(Integer, nullable=False, default=0)
    total_cost_usd = Column(Float, nullable=False, default=0.0)

    promoted_strategy_id = Column(String, nullable=True)  # set when user clicks "Promote"

    created_at = Column(DateTime, default=_utcnow, nullable=False)
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)

    __table_args__ = (
        Index("idx_re_jobs_wallet", "wallet_address"),
        Index("idx_re_jobs_status", "status"),
        Index("idx_re_jobs_created", "created_at"),
    )


class StrategyReverseEngineerIteration(Base):
    """One iteration of a reverse-engineer agent loop.

    Captures the candidate strategy code, backtest run linkage,
    scoring breakdown, the LLM's critique of the previous attempt,
    and per-iteration cost.  Every iteration is preserved (not just
    the best) so the user can replay the agent's reasoning end-to-end
    in the UI and the PDF report.
    """

    __tablename__ = "strategy_reverse_engineer_iterations"

    id = Column(String, primary_key=True)
    job_id = Column(
        String,
        ForeignKey("strategy_reverse_engineer_jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    iteration = Column(Integer, nullable=False)
    status = Column(String, nullable=False, default="running")
    # running | completed | failed

    strategy_code = Column(Text, nullable=True)
    strategy_class = Column(String, nullable=True)
    backtest_run_id = Column(String, nullable=True)

    score = Column(Float, nullable=True)
    score_breakdown_json = Column(JSON, nullable=True)
    # {trade_overlap_pct, pnl_correlation, entry_timing_mae_seconds,
    #  win_rate_actual, win_rate_backtest, ...}
    divergence_summary = Column(Text, nullable=True)
    llm_critique = Column(Text, nullable=True)
    notes = Column(Text, nullable=True)
    error = Column(Text, nullable=True)

    input_tokens = Column(Integer, nullable=False, default=0)
    output_tokens = Column(Integer, nullable=False, default=0)
    cost_usd = Column(Float, nullable=False, default=0.0)
    duration_ms = Column(Float, nullable=False, default=0.0)

    created_at = Column(DateTime, default=_utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)

    __table_args__ = (
        Index("idx_re_iter_job_iter", "job_id", "iteration"),
        UniqueConstraint("job_id", "iteration", name="uq_re_iter_job_iter"),
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


class NotifierRuntimeState(Base):
    """Durable notifier cursor and dedupe state."""

    __tablename__ = "notifier_runtime_state"

    id = Column(String, primary_key=True, default="telegram")
    close_alert_markers_json = Column(JSON, default=dict)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)


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
    latency_class = Column(String, nullable=False, default="normal")
    is_enabled = Column(Boolean, default=True)
    is_paused = Column(Boolean, default=False)
    block_new_orders = Column(Boolean, default=False, nullable=False, server_default=text("false"))
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
    last_runtime_sequence = Column(BigInteger, nullable=True)
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
    signal_id = Column(String, nullable=True, index=True)
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
    execution_wallet_address = Column(String, nullable=True, index=True)
    provider_order_id = Column(String, nullable=True, index=True)
    provider_clob_order_id = Column(String, nullable=True, index=True)
    verification_status = Column(String, nullable=False, default="local", index=True)
    verification_source = Column(String, nullable=True)
    verification_reason = Column(Text, nullable=True)
    verification_tx_hash = Column(String, nullable=True, index=True)
    verified_at = Column(DateTime, nullable=True)
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
        Index("idx_trader_orders_trader_mode_status", "trader_id", "mode", "status"),
    )


# ── Realized P&L: computed-baseline-with-verifier-override ────────
#
# ``actual_profit`` carries the BEST AVAILABLE realized P&L estimate
# at any moment.  It is populated by two layers:
#
#   1. The lifecycle / reconcile path computes ``actual_profit`` from
#      the bot's recorded fill (size, price) and the close event
#      (close_price for resolutions, sell-fill price for early exits).
#      This is the BASELINE — the UI sees real numbers immediately
#      after the close event lands, instead of $0.
#
#   2. ``polymarket_trade_verifier`` runs periodically and OVERWRITES
#      ``actual_profit`` with on-chain truth (matched against
#      transactionHash from polymarket.get_wallet_trades, or against
#      the deterministic resolution payout).  When the verifier
#      writes, it also bumps ``verification_status`` to
#      ``wallet_activity`` so the UI can flag the row as verified.
#
# The previous architecture (silent-NULL guard, removed in Fix KK)
# rejected layer 1 writes entirely — only the verifier could write
# ``actual_profit``.  This was meant to make phantom-PnL impossible,
# but in practice it produced a far worse failure mode: when the
# verifier's HTTP fetch timed out (12s budget on closed_positions),
# resolved positions sat at ``actual_profit = NULL`` for hours, and
# the UI rendered "$0 R-P&L" — operators reasonably interpreted this
# as a 100% loss when the position had actually won.  Cure was worse
# than the disease.
#
# Trade-off accepted: a buggy lifecycle computation could write a
# wrong baseline (the Athletics/Rangers case from the old guard's
# motivation: phantom +$49 when actual was -$9).  Mitigations:
#   * ``verification_status`` distinguishes baseline vs verified, so
#     dashboards can flag unverified rows.
#   * The verifier's overwrite is the durable correctness anchor —
#     even if a lifecycle write is wrong, the next verifier sweep
#     replaces it with on-chain truth.
#   * Lifecycle paths use the bot's own recorded fill data, not
#     wallet aggregates, so the conflation problem that produced the
#     old phantom case is structurally ruled out at the source.
#
# Operators relying on ``SUM(actual_profit)`` for ledger-grade P&L
# should join through ``trader_order_verification`` and filter to
# ``verification_status IN ('wallet_activity', 'manual_writeoff')``
# for verified-only aggregates.  The default (all rows) is the live
# operational P&L view.


# ── trader_order_verification: writer-isolation table ────────────────
#
# Splits the verification-related columns off TraderOrder onto a
# 1-to-1 child table.  Rationale: the Polymarket verifier and the
# orchestrator/lifecycle BOTH UPDATE the same trader_orders row.  Even
# though the verifier only touches verification_* + actual_profit and
# the orchestrator only touches status + payload_json, PG row-locks
# are at the row level, so the two writers serialize on the same
# TransactionID lock.  The 24h log captured this as
# ``LOCK CONTENTION ... UPDATE trader_orders SET status=...
# verification_status=...``.
#
# Splitting verification fields onto their own row eliminates the lock
# collision STRUCTURALLY: verifier writes to trader_order_verification,
# orchestrator writes to trader_orders, no shared row, no lock queue.
#
# Migration is staged across multiple commits:
#   * THIS commit (step 1): add table + ORM model + backfill.  No
#     application code reads or writes from the new table yet.  Schema
#     is purely additive; rolling back is just an index drop.
#   * Step 2: route verifier writes to the new table.  Orchestrator
#     keeps writing trader_orders.verification_status for the
#     wallet_position path.  Readers fall back: prefer the new table's
#     value, drop to the old column when missing.
#   * Step 3: drop old columns once readers are fully cut over.

class TraderOrderVerification(Base):
    """1:1 child of TraderOrder holding verification fields.

    Why a separate table: the verifier and the orchestrator both write
    to trader_orders today, and PG row-level locks force their writes
    to serialize on the same row even though they touch disjoint
    columns.  Moving verification fields to their own row makes the
    two writers never share a lock target.

    The DB-layer guards that used to NULL out unverified actual_profit
    were removed in Fix KK (see the comment block above the table) —
    this table now mirrors trader_orders.actual_profit verbatim, with
    verification_status carrying the "is this a verified value?" flag.
    """

    __tablename__ = "trader_order_verification"

    trader_order_id = Column(
        String,
        ForeignKey("trader_orders.id", ondelete="CASCADE"),
        primary_key=True,
    )
    verification_status = Column(String, nullable=False, default="local", index=True)
    verification_source = Column(String, nullable=True)
    verification_reason = Column(Text, nullable=True)
    verification_tx_hash = Column(String, nullable=True, index=True)
    verified_at = Column(DateTime, nullable=True)
    actual_profit = Column(Float, nullable=True)
    execution_wallet_address = Column(String, nullable=True, index=True)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow, nullable=False)

    __table_args__ = (
        Index(
            "idx_trader_order_verification_wallet_realized_pnl",
            "execution_wallet_address",
            postgresql_where=(actual_profit.isnot(None)),
        ),
    )


# The guard on trader_order_verification is intentionally scoped to this
# side table only (not trader_orders).  trader_orders.actual_profit may
# carry a lifecycle-computed baseline at any status; this table is the
# authoritative verified-P&L view and must only expose a non-null
# actual_profit when the value has been confirmed on-chain.
_VERIFIED_PNL_STATUSES = {"wallet_activity", "manual_writeoff"}


def _enforce_pnl_verification_guard_v2(mapper, connection, target):  # noqa: ANN001
    """Coerce actual_profit to None unless verification_status is verified."""
    status = str(getattr(target, "verification_status", "") or "").strip().lower()
    if status in _VERIFIED_PNL_STATUSES:
        return
    if getattr(target, "actual_profit", None) is not None:
        target.actual_profit = None


_sa_event.listen(TraderOrderVerification, "before_insert", _enforce_pnl_verification_guard_v2)
_sa_event.listen(TraderOrderVerification, "before_update", _enforce_pnl_verification_guard_v2)


# ── Dual-write mirror: TraderOrder → TraderOrderVerification ─────────
#
# Phase 3 step 2 (this commit): every successful TraderOrder write
# mirrors the verification fields onto trader_order_verification via
# a synchronous after_insert / after_update event listener.  This
# guarantees the side table is always in sync with the canonical
# columns on trader_orders WITHOUT requiring every existing caller
# (40+ direct write sites + apply_trader_order_verification +
# polymarket_trade_verifier) to be updated.
#
# Step 3 (this commit): readers prefer the side table via
# COALESCE(v.*, t.*) so once dual-write is in place the new table is
# already authoritative for new verifications.
#
# Step 4 (FUTURE commit, after validation): once readers are fully
# cut over and we've confirmed P&L numbers match in production, drop
# the verification_status / verification_source / verification_reason
# / verification_tx_hash / verified_at / actual_profit columns from
# trader_orders.  At that point the verifier and orchestrator never
# share a row-lock target.
#
# Trade-off during dual-write window: every TraderOrder UPSERT does
# an extra UPSERT on trader_order_verification in the same connection
# / same transaction.  Both UPSERTs are sub-millisecond against indexes
# on PK / trader_order_id, so the additional row-lock surface is tiny.
# The contention picture is unchanged from before this commit — that
# benefit lands at step 4.
def _mirror_trader_order_to_verification(mapper, connection, target):  # noqa: ANN001
    """Mirror verification fields onto trader_order_verification.

    Fired AFTER every insert/update of a TraderOrder row so the side
    table is always in sync.  Uses pg_insert + ON CONFLICT DO UPDATE
    so the same handler covers initial insert and ongoing updates.
    """
    from sqlalchemy.dialects.postgresql import insert as _pg_insert

    order_id = getattr(target, "id", None)
    if not order_id:
        return
    _vstatus = (getattr(target, "verification_status", None) or "local")
    _actual_profit = (
        getattr(target, "actual_profit", None)
        if _vstatus in _VERIFIED_PNL_STATUSES
        else None
    )
    payload = {
        "trader_order_id": str(order_id),
        "verification_status": _vstatus,
        "verification_source": getattr(target, "verification_source", None),
        "verification_reason": getattr(target, "verification_reason", None),
        "verification_tx_hash": getattr(target, "verification_tx_hash", None),
        "verified_at": getattr(target, "verified_at", None),
        "actual_profit": _actual_profit,
        "execution_wallet_address": getattr(target, "execution_wallet_address", None),
        "updated_at": _utcnow(),
    }
    stmt = _pg_insert(TraderOrderVerification.__table__).values(**payload)
    update_set = {k: stmt.excluded[k] for k in payload if k != "trader_order_id"}
    stmt = stmt.on_conflict_do_update(
        index_elements=["trader_order_id"],
        set_=update_set,
    )
    try:
        connection.execute(stmt)
    except Exception:
        # NEVER let a mirror failure abort the parent TraderOrder write.
        # The side table is a derived view; any lost mirror update will
        # be reconciled on the next TraderOrder write to the same row.
        pass


_sa_event.listen(TraderOrder, "after_insert", _mirror_trader_order_to_verification)
_sa_event.listen(TraderOrder, "after_update", _mirror_trader_order_to_verification)


class TraderOrderVerificationEvent(Base):
    """Immutable verification evidence attached to a trader order."""

    __tablename__ = "trader_order_verification_events"

    id = Column(String, primary_key=True)
    trader_order_id = Column(
        String,
        ForeignKey("trader_orders.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    verification_status = Column(String, nullable=False, default="local", index=True)
    source = Column(String, nullable=True, index=True)
    event_type = Column(String, nullable=False, index=True)
    reason = Column(Text, nullable=True)
    provider_order_id = Column(String, nullable=True, index=True)
    provider_clob_order_id = Column(String, nullable=True, index=True)
    execution_wallet_address = Column(String, nullable=True, index=True)
    tx_hash = Column(String, nullable=True, index=True)
    token_id = Column(String, nullable=True, index=True)
    side = Column(String, nullable=True)
    price = Column(Float, nullable=True)
    size = Column(Float, nullable=True)
    trade_timestamp = Column(DateTime, nullable=True, index=True)
    trade_id = Column(String, nullable=True, index=True)
    payload_json = Column(JSON, default=dict)
    created_at = Column(DateTime, default=_utcnow, nullable=False, index=True)

    __table_args__ = (
        Index("idx_tove_order_created", "trader_order_id", "created_at"),
        Index("idx_tove_status_created", "verification_status", "created_at"),
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
        Index("idx_trader_positions_trader_mode_status", "trader_id", "mode", "status"),
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
    # Verbosity (whisper|murmur|voice|shout) is the firehose volume tier —
    # orthogonal to severity. Only meaningful for severity='info' rows;
    # warnings/errors render regardless of the user's volume setting.
    verbosity = Column(String, nullable=True, index=True)
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


# ==================== DISCOVERY PROFILES ====================


class DiscoveryProfileTombstone(Base):
    """Permanent suppression records for seeded system discovery profiles."""

    __tablename__ = "discovery_profile_tombstones"

    slug = Column(String, primary_key=True)
    deleted_at = Column(DateTime, default=_utcnow, nullable=False)
    reason = Column(String, nullable=True)


class DiscoveryProfile(Base):
    """User-editable discovery scoring and pool selection profile.

    Each row defines a Python class extending BaseDiscoveryProfile with
    score_wallet() and select_pool() methods that control how wallets are
    ranked and how the smart pool is populated.
    """

    __tablename__ = "discovery_profiles"

    id = Column(String, primary_key=True)
    slug = Column(String, unique=True, nullable=False)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    source_code = Column(Text, nullable=False)
    class_name = Column(String, nullable=True)
    is_system = Column(Boolean, default=False, nullable=False)
    enabled = Column(Boolean, default=True)
    is_active = Column(Boolean, default=False, nullable=False)
    status = Column(String, default="unloaded")  # unloaded, loaded, error
    error_message = Column(Text, nullable=True)
    config = Column(JSON, default=dict)
    config_schema = Column(JSON, default=dict)
    profile_kind = Column(String, default="python")  # python, form
    version = Column(Integer, default=1)
    sort_order = Column(Integer, default=0)
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)

    __table_args__ = (
        Index("idx_discovery_profile_slug", "slug"),
        Index("idx_discovery_profile_enabled", "enabled"),
        Index("idx_discovery_profile_is_active", "is_active"),
        Index("idx_discovery_profile_status", "status"),
    )


class DiscoveryProfileVersion(Base):
    """Immutable discovery profile snapshots for versioned rollbacks."""

    __tablename__ = "discovery_profile_versions"

    id = Column(String, primary_key=True)
    profile_id = Column(
        String,
        ForeignKey("discovery_profiles.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    profile_slug = Column(String, nullable=False, index=True)
    version = Column(Integer, nullable=False)
    is_latest = Column(Boolean, nullable=False, default=False)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    source_code = Column(Text, nullable=False)
    class_name = Column(String, nullable=True)
    config = Column(JSON, default=dict)
    config_schema = Column(JSON, default=dict)
    profile_kind = Column(String, default="python")
    enabled = Column(Boolean, default=True)
    is_system = Column(Boolean, default=False, nullable=False)
    sort_order = Column(Integer, default=0)
    parent_version = Column(Integer, nullable=True)
    created_by = Column(String, nullable=True)
    reason = Column(Text, nullable=True)
    created_at = Column(DateTime, default=_utcnow, nullable=False)

    __table_args__ = (
        UniqueConstraint("profile_id", "version", name="uq_discovery_profile_versions_profile_version"),
        Index("idx_discovery_profile_versions_slug_version", "profile_slug", "version"),
        Index("idx_discovery_profile_versions_profile_created", "profile_id", "created_at"),
        Index("idx_discovery_profile_versions_latest", "profile_id", "is_latest"),
    )


# ==================== DATABASE SETUP ====================

_engine_kw: dict = {
    "echo": False,
    "pool_pre_ping": True,
}

_database_url = str(settings.DATABASE_URL or "").strip().lower()
if not _database_url.startswith("postgresql"):
    raise ValueError(f"DATABASE_URL must use PostgreSQL; got {settings.DATABASE_URL!r}")

# Worker subprocesses need smaller pools than the main API process but
# must have enough headroom for concurrent DB consumers (event dispatcher
# stream listener, fire-and-forget reactive tasks, demand polling, etc.).
# Postgres max_connections is set to 200 by run.ps1/run.sh.  With the
# worker process at pool_size=22 + max_overflow=12 (34 max) plus the main
# process at 20+10 (30 max), the theoretical ceiling is 64 per pair —
# well within the 200-connection budget.  Previous pool_size=18 + 8=26
# was too tight under heavy reconciliation load with many open positions.
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
_connect_args: dict = {
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
        "lock_timeout": str(max(100, int(settings.DATABASE_LOCK_TIMEOUT_MS))),
        "idle_in_transaction_session_timeout": str(max(1000, int(settings.DATABASE_IDLE_IN_TRANSACTION_TIMEOUT_MS))),
        # TCP keepalive on the server side of each connection — detect dead
        # clients within ~90 seconds (60s idle + 3×10s probes) instead of
        # relying on the OS default of 2 hours.
        "tcp_keepalives_idle": "60",
        "tcp_keepalives_interval": "10",
        "tcp_keepalives_count": "3",
        # 2026-05-09: synchronous_commit=local (was server-default ``on``).
        # The 2026-05-09 SLOW COMMIT DIAGNOSTIC captures show ``commit_ms=2077
        # dirty_rows=0`` — a no-op COMMIT taking 2 s, which is pure WAL
        # fsync latency. ``local`` keeps the local primary's WAL fsync
        # required (durable on this machine) but does NOT wait for any
        # standby acknowledgement. On a single-node deployment this is
        # equivalent to ``on`` for the *durability* guarantee but allows
        # group commit to coalesce more aggressively, dropping per-commit
        # latency from 500-2000 ms to <50 ms typical.
        #
        # Risk profile: on a hard crash during fsync, the most recent
        # commits (last few hundred ms) could be lost. For this trading
        # workload that's acceptable: orders themselves are persisted in
        # the upstream CLOB system of record (not the local DB), and
        # trade_signals/trader_decisions are recoverable from upstream
        # re-emission on restart (scanner re-finds opportunities, news
        # plane re-emits, orchestrator re-decides). If you ever attach a
        # streaming replica with hot-failover, switch back to ``on`` (or
        # ``remote_apply``) for that deployment.
        "synchronous_commit": "local",
    },
}

_engine_kw["connect_args"] = _connect_args

async_engine = create_async_engine(settings.DATABASE_URL, **_engine_kw)

_db_logger = _logging.getLogger("homerun.db.pool")
_db_logger.info(
    "Connection pool created (role=%s, pool_size=%d, max_overflow=%d)",
    "worker" if _is_worker else "main",
    _pool_size,
    _max_overflow,
)


# Enable TCP keepalive on the raw socket for every new connection.
# asyncpg does not expose a connect-time keepalive parameter, so we
# set it via a SQLAlchemy pool "connect" event which fires after the
# DBAPI connection is established.  This ensures the client side
# detects dead connections within ~90s (60s idle + 3×10s probes)
# instead of relying on the OS default of 2 hours.
@_sa_event.listens_for(async_engine.sync_engine, "connect")
def _set_tcp_keepalive(dbapi_connection, connection_record):
    import socket as _socket
    try:
        transport = getattr(dbapi_connection, "_transport", None)
        raw_sock = None
        if transport is not None:
            raw_sock = transport.get_extra_info("socket")
            if raw_sock is None:
                raw_sock = getattr(transport, "_sock", None)
        if raw_sock is None:
            protocol = getattr(dbapi_connection, "_protocol", None)
            if protocol is not None:
                t = getattr(protocol, "_transport", None)
                if t is not None:
                    raw_sock = t.get_extra_info("socket")
                    if raw_sock is None:
                        raw_sock = getattr(t, "_sock", None)
        if raw_sock is not None and isinstance(raw_sock, _socket.socket):
            raw_sock.setsockopt(_socket.SOL_SOCKET, _socket.SO_KEEPALIVE, 1)
            if hasattr(_socket, "TCP_KEEPIDLE"):
                raw_sock.setsockopt(_socket.IPPROTO_TCP, _socket.TCP_KEEPIDLE, 60)
            if hasattr(_socket, "TCP_KEEPINTVL"):
                raw_sock.setsockopt(_socket.IPPROTO_TCP, _socket.TCP_KEEPINTVL, 10)
            if hasattr(_socket, "TCP_KEEPCNT"):
                raw_sock.setsockopt(_socket.IPPROTO_TCP, _socket.TCP_KEEPCNT, 3)
    except Exception:
        pass


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
    # Weakref to the owning task so the reaper can cancel it (not just
    # invalidate the underlying connection) when checkout exceeds the
    # hard limit. Without this, the reaper closes the socket and moves
    # on, but the owning task is still scheduled — it will surface a
    # confusing InterfaceError on its next await of the (now-dead)
    # connection. Cancelling the task surfaces the error at its real
    # source. weakref so we don't pin the task in memory.
    try:
        task = asyncio.current_task()
        connection_record.info["checkout_task_ref"] = weakref.ref(task) if task is not None else None
    except RuntimeError:
        connection_record.info["checkout_task_ref"] = None
    try:
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute(
                f"SET statement_timeout = '{max(1000, int(settings.DATABASE_STATEMENT_TIMEOUT_MS))}'"
            )
            cursor.execute(
                f"SET lock_timeout = '{max(100, int(settings.DATABASE_LOCK_TIMEOUT_MS))}'"
            )
            cursor.execute(
                "SET idle_in_transaction_session_timeout = "
                f"'{max(1000, int(settings.DATABASE_IDLE_IN_TRANSACTION_TIMEOUT_MS))}'"
            )
        finally:
            cursor.close()
    except Exception:
        pass

@_sa_event.listens_for(async_engine.sync_engine, "checkin")
def _on_checkin(dbapi_connection, connection_record):
    checkout_time = connection_record.info.pop("checkout_time", None)
    checkout_task_name = connection_record.info.pop("checkout_task_name", "unknown")
    checkout_task_coro = connection_record.info.pop("checkout_task_coro", "unknown")
    if checkout_time is not None:
        elapsed = _time.monotonic() - checkout_time
        if elapsed > 30.0:
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
    # CancelledError or no-exception invalidations are virtually always
    # proactive cleanup from RetryableAsyncSession (post-cancel drain
    # plus invalidate, or pool_pre_ping failover).  They do not indicate
    # a problem with the *underlying* DB — they're cleanup signals.
    # Demote to debug so reconnect bursts don't flood the warning channel.
    is_proactive = exception is None or exception_name == "CancelledError"
    log_fn = _db_logger.debug if is_proactive else _db_logger.warning
    if checkout_time is not None:
        elapsed = _time.monotonic() - checkout_time
        log_fn(
            "Connection invalidated after %.1fs checked out (exception=%s, task=%s, coro=%s)",
            elapsed,
            exception_name,
            checkout_task_name,
            checkout_task_coro,
        )
        return
    log_fn(
        "Connection invalidated (exception=%s, task=%s, coro=%s)",
        exception_name,
        checkout_task_name,
        checkout_task_coro,
    )

AsyncSessionLocal = sessionmaker(async_engine, class_=RetryableAsyncSession, expire_on_commit=False)


# ==================== FAST-TIER ENGINE ====================
#
# Dedicated engine + session factory for the fast latency tier.  Traders with
# ``latency_class='fast'`` run through this pool exclusively; it is isolated
# from the main pool so a slow reconciliation query (30s statement_timeout on
# the main engine) cannot starve a sub-second crypto bot of connections.
#
# Deliberately small: a fast trader does one short write per trade, on an
# event-triggered cycle, so a handful of dedicated connections is plenty.  No
# overflow — we WANT ``pool_timeout`` errors surfaced fast if the pool saturates
# rather than queueing behind slow work.
#
# Statement timeout is aggressive (2500ms): any DB query the fast path makes
# that runs longer than that is pathological and should fail loud instead of
# holding a connection. Lock timeout is shorter (500ms) so a row contended
# by the scanner / orchestrator surfaces as a clean LockNotAvailable error
# (which the retry path handles cleanly) instead of letting the wait expand
# the query into the statement-timeout window — that path is the one that
# corrupts the asyncpg protocol state and produces the recurring
# "cannot switch to state X; another operation (Y) is in progress" warnings.

_FAST_POOL_SIZE = 12
_FAST_MAX_OVERFLOW = 6
_FAST_STATEMENT_TIMEOUT_MS = 2500
_FAST_LOCK_TIMEOUT_MS = 500
_FAST_IDLE_IN_TRANSACTION_TIMEOUT_MS = 3000
_FAST_POOL_TIMEOUT_SECONDS = 2

_fast_engine_kw: dict = {
    "echo": False,
    "pool_pre_ping": True,
    "pool_size": _FAST_POOL_SIZE,
    "max_overflow": _FAST_MAX_OVERFLOW,
    "pool_timeout": _FAST_POOL_TIMEOUT_SECONDS,
    "pool_recycle": max(30, int(settings.DATABASE_POOL_RECYCLE_SECONDS)),
    "pool_use_lifo": True,
}
_fast_connect_args: dict = {
    "timeout": float(max(1.0, float(settings.DATABASE_CONNECT_TIMEOUT_SECONDS))),
    "command_timeout": float(max(3.0, (_FAST_STATEMENT_TIMEOUT_MS / 1000.0) + 1.0)),
    "server_settings": {
        "timezone": "UTC",
        "statement_timeout": str(_FAST_STATEMENT_TIMEOUT_MS),
        "lock_timeout": str(_FAST_LOCK_TIMEOUT_MS),
        "idle_in_transaction_session_timeout": str(_FAST_IDLE_IN_TRANSACTION_TIMEOUT_MS),
        "tcp_keepalives_idle": "30",
        "tcp_keepalives_interval": "5",
        "tcp_keepalives_count": "3",
        # See main pool above for synchronous_commit=local rationale.
        # The fast pool serves the order-submission hot path where
        # commit latency directly extends time-to-market.
        "synchronous_commit": "local",
    },
}
_fast_engine_kw["connect_args"] = _fast_connect_args

fast_async_engine = create_async_engine(settings.DATABASE_URL, **_fast_engine_kw)
_db_logger.info(
    "Fast-tier connection pool created (pool_size=%d, max_overflow=%d, statement_timeout_ms=%d)",
    _FAST_POOL_SIZE,
    _FAST_MAX_OVERFLOW,
    _FAST_STATEMENT_TIMEOUT_MS,
)


@_sa_event.listens_for(fast_async_engine.sync_engine, "checkout")
def _fast_on_checkout(dbapi_connection, connection_record, connection_proxy):  # noqa: ANN001
    task_name, coro_name = _pool_task_context()
    connection_record.info["checkout_time"] = _time.monotonic()
    connection_record.info["checkout_task_name"] = task_name
    connection_record.info["checkout_task_coro"] = coro_name
    try:
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute(f"SET statement_timeout = '{_FAST_STATEMENT_TIMEOUT_MS}'")
            cursor.execute(f"SET lock_timeout = '{_FAST_LOCK_TIMEOUT_MS}'")
            cursor.execute(f"SET idle_in_transaction_session_timeout = '{_FAST_IDLE_IN_TRANSACTION_TIMEOUT_MS}'")
        finally:
            cursor.close()
    except Exception:
        pass


@_sa_event.listens_for(fast_async_engine.sync_engine, "checkin")
def _fast_on_checkin(dbapi_connection, connection_record):  # noqa: ANN001
    checkout_time = connection_record.info.pop("checkout_time", None)
    checkout_task_name = connection_record.info.pop("checkout_task_name", "unknown")
    checkout_task_coro = connection_record.info.pop("checkout_task_coro", "unknown")
    if checkout_time is not None:
        elapsed = _time.monotonic() - checkout_time
        # The fast pool releases connections in <100ms when the cycle is
        # purely DB-bound, but a real cycle includes a CLOB submission
        # (~300-500ms) plus DB writes; expect ~700ms-1.5s in the warm
        # path.  Warn only on genuine outliers — something on the cycle
        # is starving the loop or stalling mid-protocol.
        if elapsed > 2.0:
            _db_logger.warning(
                "Fast-tier connection held for %.2fs before return to pool (task=%s, coro=%s)",
                elapsed,
                checkout_task_name,
                checkout_task_coro,
            )


FastAsyncSessionLocal = sessionmaker(fast_async_engine, class_=RetryableAsyncSession, expire_on_commit=False)


# =====================================================================
# AUDIT POOL — isolated from operational traffic.
# =====================================================================
#
# trader_hot_state.flush_audit_buffer is a high-rate background writer
# (TraderDecision/TraderEvent/TraderDecisionCheck/TraderSignal* etc).
# Until now it shared the main pool with the orchestrator, reconciler,
# and intent-runtime projection. A single slow audit commit could
# occupy a main-pool connection for 60-115s under lock contention,
# starving everything else and triggering the cascading failure
# pattern observed in production.
#
# The audit path is non-critical — losing a batch is recoverable (re-
# queued; eventually dropped on buffer overflow). It must NEVER be
# allowed to block trading. Give it its own small pool with fast-fail
# timeouts so the worst-case audit pathology stays inside its own
# neighborhood.
#
# Pool size matches the audit task count (it's a single asyncio.Task)
# plus a tiny overflow for instrumentation reads. statement_timeout
# matches what trader_hot_state.flush_audit_buffer SET LOCALs anyway,
# so this is belt-and-suspenders.
_AUDIT_POOL_SIZE = 2
_AUDIT_MAX_OVERFLOW = 2
_AUDIT_STATEMENT_TIMEOUT_MS = 5000
_AUDIT_LOCK_TIMEOUT_MS = 2000
_AUDIT_IDLE_IN_TRANSACTION_TIMEOUT_MS = 8000

_audit_engine_kw: dict = {
    "echo": False,
    "pool_pre_ping": True,
    "pool_size": _AUDIT_POOL_SIZE,
    "max_overflow": _AUDIT_MAX_OVERFLOW,
    "pool_timeout": 5,
    "pool_recycle": max(30, int(settings.DATABASE_POOL_RECYCLE_SECONDS)),
    "pool_use_lifo": True,
}
_audit_connect_args: dict = {
    "timeout": float(max(1.0, float(settings.DATABASE_CONNECT_TIMEOUT_SECONDS))),
    "command_timeout": float((_AUDIT_STATEMENT_TIMEOUT_MS / 1000.0) + 2.0),
    "server_settings": {
        "timezone": "UTC",
        "statement_timeout": str(_AUDIT_STATEMENT_TIMEOUT_MS),
        "lock_timeout": str(_AUDIT_LOCK_TIMEOUT_MS),
        "idle_in_transaction_session_timeout": str(_AUDIT_IDLE_IN_TRANSACTION_TIMEOUT_MS),
        "tcp_keepalives_idle": "30",
        "tcp_keepalives_interval": "10",
        "tcp_keepalives_count": "3",
        # See main pool for synchronous_commit=local rationale.
        # Audit writes (decision rows, consumption upserts, cursor
        # updates) are recoverable from upstream re-emission on crash;
        # local-fsync semantics are appropriate.
        "synchronous_commit": "local",
    },
}
_audit_engine_kw["connect_args"] = _audit_connect_args

audit_async_engine = create_async_engine(settings.DATABASE_URL, **_audit_engine_kw)
_db_logger.info(
    "Audit-tier connection pool created (pool_size=%d, max_overflow=%d, statement_timeout_ms=%d)",
    _AUDIT_POOL_SIZE,
    _AUDIT_MAX_OVERFLOW,
    _AUDIT_STATEMENT_TIMEOUT_MS,
)


@_sa_event.listens_for(audit_async_engine.sync_engine, "checkout")
def _audit_on_checkout(dbapi_connection, connection_record, connection_proxy):  # noqa: ANN001
    task_name, coro_name = _pool_task_context()
    connection_record.info["checkout_time"] = _time.monotonic()
    connection_record.info["checkout_task_name"] = task_name
    connection_record.info["checkout_task_coro"] = coro_name


@_sa_event.listens_for(audit_async_engine.sync_engine, "checkin")
def _audit_on_checkin(dbapi_connection, connection_record):  # noqa: ANN001
    checkout_time = connection_record.info.pop("checkout_time", None)
    checkout_task_name = connection_record.info.pop("checkout_task_name", "unknown")
    checkout_task_coro = connection_record.info.pop("checkout_task_coro", "unknown")
    if checkout_time is not None:
        elapsed = _time.monotonic() - checkout_time
        # Audit holds connections for ms-to-low-seconds normally. Anything
        # over 8s is genuinely pathological — log loudly so we notice.
        if elapsed > 8.0:
            _db_logger.warning(
                "Audit-tier connection held for %.2fs before return to pool (task=%s, coro=%s)",
                elapsed,
                checkout_task_name,
                checkout_task_coro,
            )


AuditAsyncSessionLocal = sessionmaker(audit_async_engine, class_=RetryableAsyncSession, expire_on_commit=False)


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

_CHECKOUT_HARD_LIMIT_SECONDS = 45.0   # 45 seconds — no single checkout should take this long; previous 120s allowed cascading exhaustion
_REAPER_SCAN_INTERVAL_SECONDS = 5.0   # check more frequently to catch exhaustion earlier
_POOL_EXHAUSTION_THRESHOLD = 0.85     # trigger recovery when 85% of slots are checked out

# ==================== ZOMBIE BACKEND SWEEPER ====================
#
# Postgres backends stuck in state=active wait_event=ClientRead hold
# row-level locks indefinitely and are NOT reaped by either
# statement_timeout (the statement is already sent) or
# idle_in_transaction_session_timeout (state is not idle).  This
# happens when asyncpg is cancelled mid extended-query protocol
# (Parse/Bind without Execute/Sync).  We have seen these linger for
# 10-15+ hours, blocking every downstream write on the same rows.
#
# The client-side pool reaper cannot fix this -- the connection may
# already have been GCd on the Python side, but the server backend
# is still alive waiting for more protocol messages.  We query
# pg_stat_activity directly and pg_terminate_backend() the zombies.

_ZOMBIE_BACKEND_AGE_SECONDS = 180  # 3 minutes - generous, must not catch legit slow writes
_ZOMBIE_SWEEPER_INTERVAL_SECONDS = 30.0

_reaper_task: asyncio.Task | None = None


def _force_close_raw_socket(dbapi_conn) -> None:
    """Set a short SO_TIMEOUT on the raw socket and shut it down.

    When the ProactorEventLoop I/O transport is broken on Windows,
    socket.close() can hang indefinitely because the completion port
    callback never fires.  By calling socket.shutdown() first with a
    short timeout, we ensure the OS tears down the TCP connection at the
    kernel level regardless of the event loop state.
    """
    import socket as _socket

    try:
        transport = getattr(dbapi_conn, "_transport", None)
        raw_sock = None
        if transport is not None:
            raw_sock = getattr(transport, "_sock", None)
            if raw_sock is None:
                raw_sock = transport.get_extra_info("socket")
        if raw_sock is None:
            # asyncpg exposes the raw socket on the protocol
            protocol = getattr(dbapi_conn, "_protocol", None)
            if protocol is not None:
                transport = getattr(protocol, "_transport", None)
                if transport is not None:
                    raw_sock = getattr(transport, "_sock", None)
                    if raw_sock is None:
                        raw_sock = transport.get_extra_info("socket")
        if raw_sock is not None and isinstance(raw_sock, _socket.socket):
            try:
                raw_sock.settimeout(2.0)
                raw_sock.shutdown(_socket.SHUT_RDWR)
            except OSError:
                pass
    except Exception:
        pass


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
        #
        # On Windows with ProactorEventLoop, the I/O completion port can
        # break, causing socket.close() to hang.  Set a short OS-level
        # timeout on the raw socket first so the close cannot block
        # indefinitely.
        try:
            dbapi_conn = conn_record.dbapi_connection
            if dbapi_conn is not None:
                _force_close_raw_socket(dbapi_conn)
                conn_record.invalidate(e=TimeoutError(f"checkout exceeded {_CHECKOUT_HARD_LIMIT_SECONDS}s"))
                reaped += 1
        except Exception as e:
            _db_logger.warning("REAPER: Failed to invalidate connection: %s", e)

        # NOTE 2026-05-09: previously we ALSO called owning_task.cancel()
        # here. Removed because it was the proximate cause of the
        # "cannot switch to state 11/15; another operation (2) is in
        # progress" cascade observed in the 2026-05-09 soak (13:30:26+):
        # cancellation propagated mid-asyncpg-protocol, and the task's
        # cleanup path (rollback / finally) issued a NEW op on a
        # connection whose protocol state was still in-flight, wedging
        # multiple downstream tasks (market_cache.delete_market,
        # trader_orchestrator, trader_hot_state.flush_audit_buffer,
        # tracked_traders pool_recompute, market_data_ingestor).
        #
        # The codebase already handles InterfaceError surfacing on the
        # task's next await via RetryableSession + retry classifier
        # (see backend/utils/retry.py). Letting that path run produces
        # a clean rollback/retry instead of poisoning the protocol.
        # Keep the weakref + logged identity for diagnostics.
        task_ref = info.get("checkout_task_ref")
        owning_task = None
        if callable(task_ref):
            try:
                owning_task = task_ref()
            except Exception:
                owning_task = None
        if owning_task is not None and not owning_task.done():
            _db_logger.warning(
                "REAPER: Owning task=%s coro=%s still running after connection invalidation; "
                "letting RetryableSession surface InterfaceError on its next await",
                task_name,
                coro_name,
            )

    return reaped


async def _observe_lock_contention() -> int:
    """Probe pg_stat_activity for queries blocked >2s on row locks.

    Returns the count of blocked queries observed. Logs each one at
    WARNING with the blocking PID(s) and the query snippets so we can
    see contention BEFORE it cascades into a worker-task timeout. Pure
    diagnostic — never kills anything (the zombie sweeper handles
    actually-stuck backends).

    Uses a short-lived raw asyncpg connection separate from the
    SQLAlchemy pool so this observer can never compete with the work
    it's trying to observe.

    2026-05-09: extended to also capture the HOLDER's query + transaction
    age. Original observer logged only the blocked side ("we're waiting on
    PID X"), which left the operator guessing what X was actually doing.
    The new structured ``holders=[...]`` field on the LOCK CONTENTION line
    surfaces the holder's query, app_name, transaction age, and
    wait_event so we can identify the producer responsible (e.g. "scanner
    UPSERT trade_signals row=X holding lock 4s; orchestrator status
    UPDATE waiting").
    """
    try:
        import asyncpg
        dsn = str(settings.DATABASE_URL).replace("+asyncpg", "")
        probe_conn = await asyncio.wait_for(asyncpg.connect(dsn=dsn, timeout=3), timeout=4)
    except Exception as exc:
        _db_logger.debug("Lock observer probe connect failed: %s", exc)
        return 0
    try:
        rows = await asyncio.wait_for(
            probe_conn.fetch(
                """
                SELECT
                    pid,
                    application_name,
                    EXTRACT(EPOCH FROM now() - xact_start)::int AS age_s,
                    wait_event_type,
                    wait_event,
                    LEFT(query, 200) AS q,
                    pg_blocking_pids(pid) AS blocked_by
                FROM pg_stat_activity
                WHERE datname = current_database()
                  AND state = 'active'
                  AND wait_event_type = 'Lock'
                  AND xact_start < now() - interval '2 seconds'
                ORDER BY age_s DESC
                LIMIT 20
                """,
            ),
            timeout=4,
        )
        # Second query: for every PID that's blocking someone, capture its
        # current query + transaction state so we can name the holder in
        # the operator log. Single round-trip, bounded by the holder PID
        # set (at most 20 distinct holders given the LIMIT 20 above).
        holder_pids: set[int] = set()
        for row in rows:
            for pid in (row["blocked_by"] or []):
                if pid is not None:
                    holder_pids.add(int(pid))
        holders_by_pid: dict[int, dict[str, Any]] = {}
        if holder_pids:
            try:
                holder_rows = await asyncio.wait_for(
                    probe_conn.fetch(
                        """
                        SELECT
                            pid,
                            application_name,
                            state,
                            EXTRACT(EPOCH FROM now() - xact_start)::int AS xact_age_s,
                            EXTRACT(EPOCH FROM now() - query_start)::int AS query_age_s,
                            wait_event_type,
                            wait_event,
                            LEFT(query, 200) AS q
                        FROM pg_stat_activity
                        WHERE pid = ANY($1::int[])
                        """,
                        list(holder_pids),
                    ),
                    timeout=3,
                )
                for hr in holder_rows:
                    holders_by_pid[int(hr["pid"])] = {
                        "app": hr["application_name"] or "?",
                        "state": hr["state"] or "?",
                        "xact_age_s": int(hr["xact_age_s"] or 0),
                        "query_age_s": int(hr["query_age_s"] or 0),
                        "wait": (
                            f"{hr['wait_event_type']}/{hr['wait_event']}"
                            if hr["wait_event_type"]
                            else None
                        ),
                        "q": (hr["q"] or "").strip(),
                    }
            except Exception as exc:
                _db_logger.debug("Lock observer holder-query failed: %s", exc)
    except Exception as exc:
        _db_logger.debug("Lock observer query failed: %s", exc)
        try:
            await probe_conn.close()
        except Exception:
            pass
        return 0
    finally:
        try:
            await probe_conn.close()
        except Exception:
            pass
    if not rows:
        return 0
    # Surface contention loudly. The format is structured-friendly so
    # downstream log parsing can extract the blocking_pid and the
    # blocked-by chain. ``holders`` is per-blocked-row so each line is
    # self-contained for grep / jq.
    for row in rows:
        blocked_by = [int(p) for p in (row["blocked_by"] or []) if p is not None]
        holders_list = [holders_by_pid.get(p, {"pid": p}) for p in blocked_by]
        for h, p in zip(holders_list, blocked_by):
            h.setdefault("pid", p)
        _db_logger.warning(
            "LOCK CONTENTION blocked_pid=%d age=%ds blocked_by=%s wait=%s/%s app=%s query=%r holders=%r",
            int(row["pid"]),
            int(row["age_s"] or 0),
            blocked_by,
            row["wait_event_type"],
            row["wait_event"],
            (row["application_name"] or "?"),
            row["q"] or "",
            holders_list,
        )
    return len(rows)


async def _sweep_zombie_backends() -> int:
    """Terminate Postgres backends stuck in active Client/ClientRead.

    Returns the number of backends terminated.  These are DB-side zombies
    that the client-side pool reaper cannot see -- cancelled asyncpg
    sessions that left the server waiting for extended-protocol Execute.
    """
    # Use a short-lived raw asyncpg connection separate from the SQLAlchemy
    # pool so we never compete with the work we are trying to unblock.
    try:
        import asyncpg
        dsn = str(settings.DATABASE_URL).replace("+asyncpg", "")
        probe_conn = await asyncio.wait_for(asyncpg.connect(dsn=dsn, timeout=5), timeout=6)
    except Exception as exc:
        _db_logger.debug("Zombie sweeper probe connect failed: %s", exc)
        return 0
    terminated = 0
    try:
        rows = await asyncio.wait_for(
            probe_conn.fetch(
                """
                SELECT pid, now()-xact_start AS age, left(query, 140) AS q
                FROM pg_stat_activity
                WHERE datname = current_database()
                  AND state = 'active'
                  AND wait_event_type = 'Client'
                  AND wait_event = 'ClientRead'
                  AND xact_start < now() - make_interval(secs => $1)
                """,
                _ZOMBIE_BACKEND_AGE_SECONDS,
            ),
            timeout=5,
        )
        for row in rows:
            pid = int(row["pid"])
            try:
                ok = await asyncio.wait_for(
                    probe_conn.fetchval("SELECT pg_terminate_backend($1)", pid),
                    timeout=3,
                )
                if ok:
                    terminated += 1
                    _db_logger.error(
                        "ZOMBIE SWEEPER: terminated pid=%s age=%s query=%s",
                        pid,
                        row["age"],
                        row["q"],
                    )
            except Exception as kill_exc:
                _db_logger.warning("ZOMBIE SWEEPER: pg_terminate_backend(%s) failed: %s", pid, kill_exc)
    except Exception as query_exc:
        _db_logger.debug("ZOMBIE SWEEPER: probe query failed: %s", query_exc)
    finally:
        try:
            await asyncio.wait_for(probe_conn.close(), timeout=3)
        except Exception:
            pass
    return terminated


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

            # Phase 1b: sweep Postgres-side zombie backends (cancelled
            # extended-protocol sessions stuck in active Client/ClientRead).
            # These hold row locks indefinitely and are invisible to the
            # SQLAlchemy-side reaper.
            try:
                zombies = await _sweep_zombie_backends()
                if zombies:
                    _db_logger.error(
                        "ZOMBIE SWEEPER: terminated %d zombie backend(s)",
                        zombies,
                    )
            except Exception as sweep_exc:
                _db_logger.debug("Zombie sweeper error (non-fatal): %s", sweep_exc)

            # Phase 1c: pure-observability lock-contention probe. Logs
            # any query that has been waiting on a row lock for >2s,
            # plus the blocking PID(s). Lets us SEE contention before
            # it cascades into a worker-task timeout — useful both in
            # production logs and for post-incident diagnosis.
            try:
                contended = await _observe_lock_contention()
                if contended:
                    _db_logger.warning(
                        "LOCK OBSERVER: %d blocked-on-lock queries this scan",
                        contended,
                    )
            except Exception as obs_exc:
                _db_logger.debug("Lock observer error (non-fatal): %s", obs_exc)

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
                # After 2 consecutive scans at near-exhaustion, force pool
                # recovery — dispose all connections and start fresh.
                if consecutive_exhausted >= 2:
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
    if not getattr(session, "is_active", True):
        yield
        return
    try:
        dirty = getattr(session, "dirty", None)
        new = getattr(session, "new", None)
        if dirty or new:
            await session.flush()
    except Exception:
        pass
    try:
        reset = getattr(session, "reset", None)
        if callable(reset):
            await reset()
    except Exception:
        pass
    try:
        yield
    finally:
        # No action needed — the session lazily reconnects on next use.
        pass


def _run_alembic_upgrade(connection) -> None:
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import inspect as _sa_inspect

    backend_root = Path(__file__).resolve().parents[1]
    alembic_cfg = Config(str(backend_root / "alembic.ini"))
    alembic_cfg.set_main_option("script_location", str(backend_root / "alembic"))
    alembic_cfg.set_main_option("sqlalchemy.url", str(connection.engine.url))
    alembic_cfg.attributes["connection"] = connection

    # Fresh-DB bootstrap path.  The historical baseline migration uses
    # ``Base.metadata.create_all`` — i.e. it dynamically reflects the
    # CURRENT SQLAlchemy model, not the schema frozen at baseline-author
    # time.  Sixty-plus post-baseline ``op.add_column`` migrations
    # therefore collide on a from-empty replay because the columns they
    # try to add already exist (create_all built them from the latest
    # model).  The two-phase fix here: build the schema from the
    # current model, then stamp head so the runner skips the broken
    # add_column chain.  Existing DBs (with ``alembic_version``) take
    # the normal upgrade branch — zero behavior change.
    if not _sa_inspect(connection).has_table("alembic_version"):
        Base.metadata.create_all(bind=connection)
        command.stamp(alembic_cfg, "head")
        return

    command.upgrade(alembic_cfg, "head")


async def init_database():
    """Initialize database and apply Alembic migrations.

    Retries the initial connection up to 30 times (≈30s) to handle the
    race where PostgreSQL is still starting when the backend launches.
    """
    import asyncio as _asyncio
    import logging as _logging

    _log = _logging.getLogger(__name__)

    from models.model_registry import register_all_models

    register_all_models()

    max_retries = 30
    for attempt in range(max_retries):
        try:
            async with async_engine.begin() as conn:
                await conn.run_sync(_run_alembic_upgrade)
            return
        except Exception as exc:
            if attempt >= max_retries - 1:
                raise
            _log.warning(
                "Database initialization failed (attempt %d/%d), retrying in 1s",
                attempt + 1,
                max_retries,
                exc_info=exc,
            )
            await _asyncio.sleep(1)


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
        if session.in_transaction():
            try:
                await session.rollback()
            except Exception:
                try:
                    await session.invalidate()
                except Exception:
                    pass
        # close() is guaranteed to never raise (it handles
        # CancelledError and falls back to invalidate internally).
        await session.close()
