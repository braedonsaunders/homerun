import asyncio
import random
import uuid
from datetime import datetime
from utils.utcnow import utcnow, utcfromtimestamp
from typing import Optional
from sqlalchemy import select, and_

from models.database import (
    CopyTradingConfig,
    CopyTradingMode,
    CopiedTrade,
    TrackedWallet,
    SimulationAccount,
    SimulationPosition,
    SimulationTrade,
    TradeStatus,
    PositionSide,
    AsyncSessionLocal,
)
from models.opportunity import ArbitrageOpportunity
from services.polymarket import polymarket_client
from services.pause_state import global_pause_state
from services.depth_analyzer import depth_analyzer
from services.token_circuit_breaker import token_circuit_breaker
from utils.logger import get_logger
from config import settings as app_settings

logger = get_logger("copy_trader")

MIN_WHALE_SHARES = (
    app_settings.MIN_WHALE_SHARES
)  # Ignore noise trades below this threshold
MIN_CASH_VALUE = 1.01  # Minimum cash value for order execution

# Maximum size of in-memory dedup cache before eviction
_DEDUP_CACHE_MAX = 10_000


class CopyTradingService:
    """Full copy-trading service that mirrors trades from source wallets.

    Supports two modes:
    - ALL_TRADES: Mirrors every buy/sell from the source wallet
    - ARB_ONLY: Only copies trades matching detected arbitrage opportunities

    Features:
    - Real-time WebSocket trade detection (<1s latency)
    - Direct event processing (bypasses HTTP API for WS events)
    - In-memory deduplication for instant dedup (no DB round-trip)
    - Trade deduplication by source trade ID
    - Proportional position sizing
    - Buy and sell mirroring
    - Both simulation and live execution paths
    - Source wallet position tracking
    """

    def __init__(self):
        self._running = False
        self._poll_interval = (
            120  # seconds (fallback only; WS monitor provides real-time)
        )
        self._active_configs: dict[str, CopyTradingConfig] = {}
        # In-memory cache of source wallet positions for diffing
        self._wallet_positions: dict[str, list[dict]] = {}
        # In-memory dedup cache: tx_hash -> timestamp for instant dedup
        # Prevents DB round-trips on the hot path for real-time events
        self._realtime_dedup_cache: dict[str, datetime] = {}
        # Broadcast callback for pushing copy trade events to frontend via WS
        self._ws_broadcast_callback = None

    # ==================== CONFIG MANAGEMENT ====================

    async def add_copy_config(
        self,
        source_wallet: str,
        account_id: str,
        copy_mode: str = "all_trades",
        min_roi_threshold: float = 2.5,
        max_position_size: float = 1000.0,
        copy_delay_seconds: int = 5,
        slippage_tolerance: float = 1.0,
        proportional_sizing: bool = False,
        proportional_multiplier: float = 1.0,
        copy_buys: bool = True,
        copy_sells: bool = True,
        market_categories: list[str] = None,
    ) -> CopyTradingConfig:
        """Add a wallet to copy trade"""
        async with AsyncSessionLocal() as session:
            # Verify account exists
            account = await session.get(SimulationAccount, account_id)
            if not account:
                raise ValueError(f"Account not found: {account_id}")

            mode = CopyTradingMode(copy_mode)

            config = CopyTradingConfig(
                id=str(uuid.uuid4()),
                source_wallet=source_wallet.lower(),
                account_id=account_id,
                enabled=True,
                copy_mode=mode,
                min_roi_threshold=min_roi_threshold,
                max_position_size=max_position_size,
                copy_delay_seconds=copy_delay_seconds,
                slippage_tolerance=slippage_tolerance,
                proportional_sizing=proportional_sizing,
                proportional_multiplier=proportional_multiplier,
                copy_buys=copy_buys,
                copy_sells=copy_sells,
                market_categories=market_categories or [],
            )
            session.add(config)

            # Also ensure wallet is tracked
            wallet = await session.get(TrackedWallet, source_wallet.lower())
            if not wallet:
                wallet = TrackedWallet(
                    address=source_wallet.lower(), label="Copy Target"
                )
                session.add(wallet)

            await session.commit()
            await session.refresh(config)

            self._active_configs[config.id] = config

            # Keep WS monitor membership in sync when running.
            if self._running:
                try:
                    from services.wallet_ws_monitor import wallet_ws_monitor

                    wallet_ws_monitor.add_wallet(source_wallet, source="copy_trader")
                except Exception:
                    pass

            logger.info(
                "Added copy trading config",
                config_id=config.id,
                source_wallet=source_wallet,
                account_id=account_id,
                copy_mode=copy_mode,
            )

            return config

    async def remove_copy_config(self, config_id: str):
        """Remove a copy trading configuration"""
        async with AsyncSessionLocal() as session:
            config = await session.get(CopyTradingConfig, config_id)
            if config:
                await session.delete(config)
                await session.commit()
                self._active_configs.pop(config_id, None)

                if self._running:
                    try:
                        from services.wallet_ws_monitor import wallet_ws_monitor

                        remaining = [
                            c.source_wallet
                            for c in self._active_configs.values()
                            if c.enabled
                        ]
                        wallet_ws_monitor.set_wallets_for_source(
                            "copy_trader", remaining
                        )
                    except Exception:
                        pass

                logger.info("Removed copy trading config", config_id=config_id)

    async def get_configs(
        self, account_id: Optional[str] = None
    ) -> list[CopyTradingConfig]:
        """Get all copy trading configurations"""
        async with AsyncSessionLocal() as session:
            query = select(CopyTradingConfig)
            if account_id:
                query = query.where(CopyTradingConfig.account_id == account_id)

            result = await session.execute(query)
            return list(result.scalars().all())

    async def enable_config(self, config_id: str, enabled: bool):
        """Enable or disable a copy trading configuration"""
        async with AsyncSessionLocal() as session:
            config = await session.get(CopyTradingConfig, config_id)
            if config:
                config.enabled = enabled
                await session.commit()

                logger.info(
                    "Updated copy trading config", config_id=config_id, enabled=enabled
                )

    async def update_config(self, config_id: str, **kwargs):
        """Update arbitrary fields on a copy trading configuration"""
        async with AsyncSessionLocal() as session:
            config = await session.get(CopyTradingConfig, config_id)
            if not config:
                raise ValueError(f"Config not found: {config_id}")

            allowed_fields = {
                "enabled",
                "copy_mode",
                "min_roi_threshold",
                "max_position_size",
                "copy_delay_seconds",
                "slippage_tolerance",
                "proportional_sizing",
                "proportional_multiplier",
                "copy_buys",
                "copy_sells",
                "market_categories",
            }

            for key, value in kwargs.items():
                if key in allowed_fields and value is not None:
                    if key == "copy_mode":
                        value = CopyTradingMode(value)
                    setattr(config, key, value)

            await session.commit()
            await session.refresh(config)

            # Update in-memory cache
            self._active_configs[config_id] = config

            if self._running:
                try:
                    from services.wallet_ws_monitor import wallet_ws_monitor

                    remaining = [
                        c.source_wallet
                        for c in self._active_configs.values()
                        if c.enabled
                    ]
                    wallet_ws_monitor.set_wallets_for_source("copy_trader", remaining)
                except Exception:
                    pass

            logger.info(
                "Updated copy trading config",
                config_id=config_id,
                fields=list(kwargs.keys()),
            )
            return config

    def set_ws_broadcast(self, callback):
        """Set the WebSocket broadcast callback for pushing copy trade events to frontend.

        Args:
            callback: Async callable that accepts a dict message to broadcast.
        """
        self._ws_broadcast_callback = callback

    async def _broadcast_copy_event(self, event_type: str, data: dict):
        """Broadcast a copy trading event to connected frontend clients."""
        if self._ws_broadcast_callback:
            try:
                await self._ws_broadcast_callback(
                    {"type": event_type, "data": data}
                )
            except Exception as e:
                logger.error("Failed to broadcast copy event", error=str(e))

    def _dedup_check_and_mark(self, dedup_key: str) -> bool:
        """Check if a trade key was already processed (in-memory, no DB).

        Returns True if this is a DUPLICATE (already seen).
        Marks it as seen if new.
        """
        if dedup_key in self._realtime_dedup_cache:
            return True  # Already processed

        # Evict old entries if cache is too large
        if len(self._realtime_dedup_cache) > _DEDUP_CACHE_MAX:
            # Remove oldest half
            sorted_keys = sorted(
                self._realtime_dedup_cache.keys(),
                key=lambda k: self._realtime_dedup_cache[k],
            )
            for k in sorted_keys[: _DEDUP_CACHE_MAX // 2]:
                del self._realtime_dedup_cache[k]

        self._realtime_dedup_cache[dedup_key] = utcnow()
        return False

    # ==================== REAL-TIME EVENT PROCESSING ====================

    async def _process_realtime_event(self, event, config: CopyTradingConfig):
        """Process a single real-time WalletTradeEvent directly, bypassing HTTP API.

        This is the fast path: the WalletTradeEvent already contains all trade
        data (token_id, side, size, price, tx_hash) from on-chain event parsing.
        No HTTP API call is needed.

        Target latency: <500ms from block detection to copy execution.
        """
        start_time = utcnow()

        # Build a trade dict from the WalletTradeEvent fields
        trade = {
            "id": f"ws_{event.tx_hash}_{event.token_id}",
            "side": event.side,
            "price": event.price,
            "size": event.size,
            "asset": event.token_id,
            "assetId": event.token_id,
            "market": "",  # Not available from on-chain event
            "condition_id": "",
            "outcome": "",
            "title": "",
            "question": "",
            "timestamp": event.timestamp.isoformat() if event.timestamp else None,
            "matchTime": event.timestamp.isoformat() if event.timestamp else None,
            "_realtime": True,
            "_tx_hash": event.tx_hash,
            "_detection_latency_ms": event.latency_ms,
        }

        # In-memory dedup (no DB round-trip)
        dedup_key = f"{config.id}:{trade['id']}"
        if self._dedup_check_and_mark(dedup_key):
            logger.debug(
                "Realtime dedup: already processed",
                tx_hash=event.tx_hash,
                config_id=config.id,
            )
            return

        # Check if we should copy this trade
        should_copy, reason = self._should_copy_trade(trade, config)
        if not should_copy:
            logger.debug(
                "Realtime skip",
                reason=reason,
                tx_hash=event.tx_hash,
            )
            # Record as skipped for audit trail (fire-and-forget to DB)
            asyncio.create_task(
                self._record_copied_trade(
                    config,
                    source_trade_id=trade["id"],
                    market_id="",
                    market_question="",
                    token_id=event.token_id,
                    side=event.side,
                    outcome="",
                    source_price=event.price,
                    source_size=event.size,
                    source_timestamp=event.timestamp,
                    status="skipped",
                    error=reason,
                )
            )
            return

        # In ARB_ONLY mode, check for matching opportunity
        if config.copy_mode == CopyTradingMode.ARB_ONLY:
            opp = await self._check_arb_match(trade)
            if not opp:
                return
            if opp.roi_percent < config.min_roi_threshold:
                return

        # Execute the copy — NO intentional delay for real-time events
        # (copy_delay_seconds is skipped to achieve <500ms execution)
        side = event.side.upper()
        if side == "BUY":
            result = await self._execute_copy_buy_realtime(trade, config, event)
        elif side == "SELL":
            result = await self._execute_copy_sell(trade, config)
        else:
            return

        # Calculate total pipeline latency
        execution_time = utcnow()
        pipeline_latency_ms = (execution_time - start_time).total_seconds() * 1000.0
        total_latency_ms = (
            event.latency_ms + pipeline_latency_ms
        )  # detection + processing

        logger.info(
            "REALTIME copy trade complete",
            config_id=config.id,
            side=side,
            token_id=event.token_id,
            size=event.size,
            price=event.price,
            detection_latency_ms=round(event.latency_ms, 1),
            pipeline_latency_ms=round(pipeline_latency_ms, 1),
            total_latency_ms=round(total_latency_ms, 1),
            tx_hash=event.tx_hash,
        )

        # Broadcast to frontend via WebSocket
        await self._broadcast_copy_event(
            "copy_trade_executed",
            {
                "config_id": config.id,
                "source_wallet": config.source_wallet,
                "side": side,
                "token_id": event.token_id,
                "source_price": event.price,
                "source_size": event.size,
                "tx_hash": event.tx_hash,
                "detection_latency_ms": round(event.latency_ms, 1),
                "pipeline_latency_ms": round(pipeline_latency_ms, 1),
                "total_latency_ms": round(total_latency_ms, 1),
                "status": "executed" if result else "failed",
                "executed_at": execution_time.isoformat(),
            },
        )

    async def _execute_copy_buy_realtime(
        self,
        trade: dict,
        config: CopyTradingConfig,
        event,
    ) -> Optional[CopiedTrade]:
        """Fast-path BUY execution for real-time events.

        Skips copy_delay_seconds to minimize latency. Uses the same
        execution logic as _execute_copy_buy but without the intentional delay
        and with streamlined error handling.
        """
        source_price = event.price
        source_size = event.size
        token_id = event.token_id
        trade_id = trade["id"]

        # Filter out noise trades below minimum whale size
        if source_size < MIN_WHALE_SHARES:
            logger.debug(
                f"Realtime: skipping small trade: {source_size} shares < {MIN_WHALE_SHARES}"
            )
            asyncio.create_task(
                self._record_copied_trade(
                    config, trade_id, "", "", token_id, "BUY", "",
                    source_price, source_size, event.timestamp,
                    status="skipped",
                    error=f"Below minimum whale size ({source_size} < {MIN_WHALE_SHARES})",
                )
            )
            return None

        # Get account to check capital
        async with AsyncSessionLocal() as session:
            account = await session.get(SimulationAccount, config.account_id)
            if not account:
                return None

            copy_size = self._calculate_copy_size(trade, config, account.current_capital)
            if copy_size <= 0:
                return None

            # Probabilistic sub-minimum execution
            if copy_size * source_price < MIN_CASH_VALUE and copy_size > 0:
                import random
                prob = (copy_size * source_price) / MIN_CASH_VALUE
                if random.random() < prob:
                    copy_size = MIN_CASH_VALUE / source_price
                else:
                    return None

        # Per-token circuit breaker
        if token_id:
            is_tripped, trip_reason = token_circuit_breaker.is_tripped(token_id)
            if is_tripped:
                return None

        # Depth check
        if token_id:
            try:
                depth_result = await depth_analyzer.check_depth(
                    token_id=token_id,
                    side="BUY",
                    target_price=source_price,
                    required_size_usd=source_price * copy_size,
                    trade_context="copy_trader_realtime",
                )
                if not depth_result.has_sufficient_depth:
                    token_circuit_breaker.trip_token(
                        token_id, "insufficient_depth_realtime",
                        {"available": depth_result.available_depth_usd},
                    )
                    return None
            except Exception as e:
                logger.error("Realtime depth check failed", error=str(e))

        # Record in circuit breaker
        if token_id:
            token_circuit_breaker.record_trade(token_id, copy_size, source_price, "BUY")

        # NO copy_delay — execute immediately for real-time
        # Get current price to check slippage
        try:
            if token_id:
                current_price = await polymarket_client.get_price(token_id, side="BUY")
            else:
                current_price = source_price
        except Exception:
            current_price = source_price

        # Check slippage tolerance
        if current_price > 0 and source_price > 0:
            slippage_pct = abs(current_price - source_price) / source_price * 100
            if slippage_pct > config.slippage_tolerance:
                asyncio.create_task(
                    self._record_copied_trade(
                        config, trade_id, "", "", token_id, "BUY", "",
                        source_price, source_size, event.timestamp,
                        status="skipped",
                        error=f"Slippage {slippage_pct:.1f}% exceeds tolerance {config.slippage_tolerance}%",
                    )
                )
                return None

        execution_price = current_price if current_price > 0 else source_price

        # Execute the BUY in simulation
        try:
            sim_trade = await self._execute_sim_buy(
                config.account_id,
                market_id="",
                market_question=f"RT copy: {event.tx_hash[:16]}",
                token_id=token_id,
                outcome="",
                price=execution_price,
                size=copy_size,
                copied_from=config.source_wallet,
            )

            copied = await self._record_copied_trade(
                config, trade_id, "", "",
                token_id, "BUY", "",
                source_price, source_size, event.timestamp,
                status="executed",
                executed_price=execution_price,
                executed_size=copy_size,
                simulation_trade_id=sim_trade.id,
            )

            # Update config stats
            async with AsyncSessionLocal() as session:
                db_config = await session.get(CopyTradingConfig, config.id)
                if db_config:
                    db_config.total_copied += 1
                    db_config.successful_copies += 1
                    db_config.total_buys_copied += 1
                    await session.commit()

            return copied

        except Exception as e:
            logger.error("Realtime copy BUY failed", error=str(e))
            async with AsyncSessionLocal() as session:
                db_config = await session.get(CopyTradingConfig, config.id)
                if db_config:
                    db_config.total_copied += 1
                    db_config.failed_copies += 1
                    await session.commit()
            return None

    # ==================== TRADE DETECTION ====================

    async def _get_new_trades(
        self,
        wallet_address: str,
        config: CopyTradingConfig,
    ) -> list[dict]:
        """Fetch new trades from source wallet since last processed trade.

        Returns trades sorted oldest-first for sequential processing.
        """
        try:
            trades = await polymarket_client.get_wallet_trades(
                wallet_address, limit=100
            )

            if not trades:
                return []

            # Deduplicate: skip trades we've already copied
            new_trades = []
            async with AsyncSessionLocal() as session:
                for trade in trades:
                    trade_id = trade.get("id", "")
                    if not trade_id:
                        continue

                    # Check if this trade ID was already processed for this config
                    existing = await session.execute(
                        select(CopiedTrade).where(
                            and_(
                                CopiedTrade.config_id == config.id,
                                CopiedTrade.source_trade_id == trade_id,
                            )
                        )
                    )
                    if existing.scalar_one_or_none():
                        continue

                    new_trades.append(trade)

            # Return oldest first so we process in chronological order
            new_trades.reverse()
            return new_trades

        except Exception as e:
            logger.error(
                "Error fetching trades for wallet", wallet=wallet_address, error=str(e)
            )
            return []

    def _should_copy_trade(
        self, trade: dict, config: CopyTradingConfig
    ) -> tuple[bool, str]:
        """Determine whether a trade should be copied based on config filters.

        Returns (should_copy, reason) tuple.
        """
        side = (trade.get("side") or "").upper()
        price = float(trade.get("price", 0) or 0)
        size = float(trade.get("size", 0) or trade.get("amount", 0) or 0)

        # Filter by trade direction
        if side == "BUY" and not config.copy_buys:
            return False, "buy copying disabled"
        if side == "SELL" and not config.copy_sells:
            return False, "sell copying disabled"

        # Filter out zero-value trades
        if price <= 0 or size <= 0:
            return False, "zero price or size"

        return True, "ok"

    async def _check_arb_match(self, trade: dict) -> Optional[ArbitrageOpportunity]:
        """Check if a trade matches a detected arbitrage opportunity (arb_only mode)."""
        from models.database import AsyncSessionLocal
        from services import shared_state

        trade_market = trade.get("market", trade.get("condition_id", ""))
        async with AsyncSessionLocal() as session:
            opportunities = await shared_state.get_opportunities_from_db(session, None)

        for opp in opportunities:
            for market in opp.markets:
                if market.get("id") == trade_market:
                    return opp

        return None

    # ==================== POSITION SIZING ====================

    def _calculate_copy_size(
        self,
        source_trade: dict,
        config: CopyTradingConfig,
        account_capital: float,
    ) -> float:
        """Calculate the position size for a copy trade.

        Supports:
        - Fixed max: caps at max_position_size
        - Proportional: scales by proportional_multiplier relative to source size
        """
        source_size = float(
            source_trade.get("size", 0) or source_trade.get("amount", 0) or 0
        )
        source_price = float(source_trade.get("price", 0) or 0)
        source_cost = source_size * source_price

        if config.proportional_sizing:
            # Scale source position by multiplier
            target_cost = source_cost * config.proportional_multiplier
        else:
            # Use source size directly, capped at max
            target_cost = source_cost

        # Apply maximum position size cap
        target_cost = min(target_cost, config.max_position_size)

        # Don't exceed available capital (leave 1% buffer)
        max_from_capital = account_capital * 0.99
        target_cost = min(target_cost, max_from_capital)

        # Convert back to shares at the current price
        if source_price > 0:
            target_size = target_cost / source_price
        else:
            target_size = 0.0

        return target_size

    # ==================== TRADE EXECUTION ====================

    async def _execute_copy_buy(
        self,
        trade: dict,
        config: CopyTradingConfig,
    ) -> CopiedTrade:
        """Execute a copy of a BUY trade in simulation."""
        source_price = float(trade.get("price", 0) or 0)
        source_size = float(trade.get("size", 0) or trade.get("amount", 0) or 0)
        market_id = trade.get("market", trade.get("condition_id", ""))
        token_id = trade.get("asset", trade.get("assetId", ""))
        outcome = trade.get("outcome", "")
        market_question = trade.get("title", trade.get("question", market_id))
        trade_id = trade.get("id", str(uuid.uuid4()))
        source_ts_raw = trade.get("timestamp", trade.get("matchTime", ""))

        source_timestamp = None
        if source_ts_raw:
            try:
                if isinstance(source_ts_raw, (int, float)):
                    source_timestamp = utcfromtimestamp(source_ts_raw)
                else:
                    source_timestamp = datetime.fromisoformat(
                        str(source_ts_raw).replace("Z", "+00:00")
                    )
            except (ValueError, OSError):
                source_timestamp = None

        # Filter out noise trades below minimum whale size
        if source_size < MIN_WHALE_SHARES:
            logger.debug(
                f"Skipping small whale trade: {source_size} shares < {MIN_WHALE_SHARES} minimum"
            )
            return await self._record_copied_trade(
                config,
                trade_id,
                market_id,
                market_question,
                token_id,
                "BUY",
                outcome,
                source_price,
                source_size,
                source_timestamp,
                status="skipped",
                error=f"Below minimum whale size ({source_size} < {MIN_WHALE_SHARES})",
            )

        # Get account to check capital
        async with AsyncSessionLocal() as session:
            account = await session.get(SimulationAccount, config.account_id)
            if not account:
                return await self._record_copied_trade(
                    config,
                    trade_id,
                    market_id,
                    market_question,
                    token_id,
                    "BUY",
                    outcome,
                    source_price,
                    source_size,
                    source_timestamp,
                    status="failed",
                    error="Account not found",
                )

            copy_size = self._calculate_copy_size(
                trade, config, account.current_capital
            )

            if copy_size <= 0:
                return await self._record_copied_trade(
                    config,
                    trade_id,
                    market_id,
                    market_question,
                    token_id,
                    "BUY",
                    outcome,
                    source_price,
                    source_size,
                    source_timestamp,
                    status="skipped",
                    error="Insufficient capital or zero size",
                )

            # Probabilistic sub-minimum execution: if copy value is below
            # the $1.01 minimum order size, randomly decide whether to
            # round up to the minimum or skip entirely.
            if copy_size * source_price < MIN_CASH_VALUE and copy_size > 0:
                prob = (copy_size * source_price) / MIN_CASH_VALUE
                if random.random() < prob:
                    copy_size = MIN_CASH_VALUE / source_price  # Execute at minimum
                    logger.info(
                        "Probabilistic bump to minimum order size",
                        original_value=copy_size * source_price,
                        prob=round(prob, 2),
                    )
                else:
                    return await self._record_copied_trade(
                        config,
                        trade_id,
                        market_id,
                        market_question,
                        token_id,
                        "BUY",
                        outcome,
                        source_price,
                        source_size,
                        source_timestamp,
                        status="skipped",
                        error=f"Probabilistic skip (p={prob:.2f})",
                    )

        # Check per-token circuit breaker before executing
        if token_id:
            is_tripped, trip_reason = token_circuit_breaker.is_tripped(token_id)
            if is_tripped:
                return await self._record_copied_trade(
                    config,
                    trade_id,
                    market_id,
                    market_question,
                    token_id,
                    "BUY",
                    outcome,
                    source_price,
                    source_size,
                    source_timestamp,
                    status="skipped",
                    error=f"Token tripped: {trip_reason}",
                )

        # Depth check before executing
        if token_id:
            try:
                depth_result = await depth_analyzer.check_depth(
                    token_id=token_id,
                    side="BUY",
                    target_price=source_price,
                    required_size_usd=source_price * copy_size,
                    trade_context="copy_trader",
                )
                if not depth_result.has_sufficient_depth:
                    token_circuit_breaker.trip_token(
                        token_id,
                        "insufficient_depth_copy",
                        {"available": depth_result.available_depth_usd},
                    )
                    return await self._record_copied_trade(
                        config,
                        trade_id,
                        market_id,
                        market_question,
                        token_id,
                        "BUY",
                        outcome,
                        source_price,
                        source_size,
                        source_timestamp,
                        status="skipped",
                        error=f"Insufficient depth: ${depth_result.available_depth_usd:.0f}",
                    )
            except Exception as e:
                logger.error("Depth check failed in copy trader", error=str(e))

        # Record trade in token circuit breaker for trip detection
        if token_id:
            token_circuit_breaker.record_trade(token_id, copy_size, source_price, "BUY")

        # Wait configured delay before executing
        if config.copy_delay_seconds > 0:
            await asyncio.sleep(config.copy_delay_seconds)

        # Get current price to check slippage
        try:
            if token_id:
                current_price = await polymarket_client.get_price(token_id, side="BUY")
            else:
                current_price = source_price
        except Exception:
            current_price = source_price

        # Check slippage tolerance
        if current_price > 0 and source_price > 0:
            slippage_pct = abs(current_price - source_price) / source_price * 100
            if slippage_pct > config.slippage_tolerance:
                return await self._record_copied_trade(
                    config,
                    trade_id,
                    market_id,
                    market_question,
                    token_id,
                    "BUY",
                    outcome,
                    source_price,
                    source_size,
                    source_timestamp,
                    status="skipped",
                    error=f"Slippage {slippage_pct:.1f}% exceeds tolerance {config.slippage_tolerance}%",
                )

        execution_price = current_price if current_price > 0 else source_price

        # Execute the BUY in simulation
        try:
            sim_trade = await self._execute_sim_buy(
                config.account_id,
                market_id=market_id,
                market_question=market_question,
                token_id=token_id,
                outcome=outcome,
                price=execution_price,
                size=copy_size,
                copied_from=config.source_wallet,
            )

            copied = await self._record_copied_trade(
                config,
                trade_id,
                market_id,
                market_question,
                token_id,
                "BUY",
                outcome,
                source_price,
                source_size,
                source_timestamp,
                status="executed",
                executed_price=execution_price,
                executed_size=copy_size,
                simulation_trade_id=sim_trade.id,
            )

            # Update config stats
            async with AsyncSessionLocal() as session:
                db_config = await session.get(CopyTradingConfig, config.id)
                if db_config:
                    db_config.total_copied += 1
                    db_config.successful_copies += 1
                    db_config.total_buys_copied += 1
                    await session.commit()

            logger.info(
                "Copied BUY trade",
                config_id=config.id,
                source_wallet=config.source_wallet,
                market=market_question[:60] if market_question else market_id,
                outcome=outcome,
                source_price=source_price,
                exec_price=execution_price,
                size=copy_size,
            )

            return copied

        except Exception as e:
            logger.error("Failed to execute copy BUY", error=str(e))
            async with AsyncSessionLocal() as session:
                db_config = await session.get(CopyTradingConfig, config.id)
                if db_config:
                    db_config.total_copied += 1
                    db_config.failed_copies += 1
                    await session.commit()

            return await self._record_copied_trade(
                config,
                trade_id,
                market_id,
                market_question,
                token_id,
                "BUY",
                outcome,
                source_price,
                source_size,
                source_timestamp,
                status="failed",
                error=str(e),
            )

    async def _execute_copy_sell(
        self,
        trade: dict,
        config: CopyTradingConfig,
    ) -> CopiedTrade:
        """Execute a copy of a SELL trade by closing matching simulation positions."""
        source_price = float(trade.get("price", 0) or 0)
        source_size = float(trade.get("size", 0) or trade.get("amount", 0) or 0)
        market_id = trade.get("market", trade.get("condition_id", ""))
        token_id = trade.get("asset", trade.get("assetId", ""))
        outcome = trade.get("outcome", "")
        market_question = trade.get("title", trade.get("question", market_id))
        trade_id = trade.get("id", str(uuid.uuid4()))
        source_ts_raw = trade.get("timestamp", trade.get("matchTime", ""))

        source_timestamp = None
        if source_ts_raw:
            try:
                if isinstance(source_ts_raw, (int, float)):
                    source_timestamp = utcfromtimestamp(source_ts_raw)
                else:
                    source_timestamp = datetime.fromisoformat(
                        str(source_ts_raw).replace("Z", "+00:00")
                    )
            except (ValueError, OSError):
                source_timestamp = None

        # Wait configured delay
        if config.copy_delay_seconds > 0:
            await asyncio.sleep(config.copy_delay_seconds)

        # Find matching open positions in our simulation account
        try:
            pnl = await self._close_sim_position(
                config.account_id,
                market_id=market_id,
                token_id=token_id,
                outcome=outcome,
                sell_price=source_price,
                copied_from=config.source_wallet,
            )

            copied = await self._record_copied_trade(
                config,
                trade_id,
                market_id,
                market_question,
                token_id,
                "SELL",
                outcome,
                source_price,
                source_size,
                source_timestamp,
                status="executed",
                executed_price=source_price,
                executed_size=source_size,
                realized_pnl=pnl,
            )

            # Update config stats
            async with AsyncSessionLocal() as session:
                db_config = await session.get(CopyTradingConfig, config.id)
                if db_config:
                    db_config.total_copied += 1
                    db_config.successful_copies += 1
                    db_config.total_sells_copied += 1
                    db_config.total_pnl += pnl or 0.0
                    await session.commit()

            logger.info(
                "Copied SELL trade",
                config_id=config.id,
                source_wallet=config.source_wallet,
                market=market_question[:60] if market_question else market_id,
                outcome=outcome,
                price=source_price,
                pnl=pnl,
            )

            return copied

        except Exception as e:
            logger.error("Failed to execute copy SELL", error=str(e))
            async with AsyncSessionLocal() as session:
                db_config = await session.get(CopyTradingConfig, config.id)
                if db_config:
                    db_config.total_copied += 1
                    db_config.failed_copies += 1
                    await session.commit()

            return await self._record_copied_trade(
                config,
                trade_id,
                market_id,
                market_question,
                token_id,
                "SELL",
                outcome,
                source_price,
                source_size,
                source_timestamp,
                status="failed",
                error=str(e),
            )

    # ==================== SIMULATION EXECUTION ====================

    async def _execute_sim_buy(
        self,
        account_id: str,
        market_id: str,
        market_question: str,
        token_id: str,
        outcome: str,
        price: float,
        size: float,
        copied_from: str,
    ) -> SimulationTrade:
        """Execute a buy trade in the simulation account."""
        total_cost = price * size

        async with AsyncSessionLocal() as session:
            account = await session.get(SimulationAccount, account_id)
            if not account:
                raise ValueError(f"Account not found: {account_id}")

            if total_cost > account.current_capital:
                raise ValueError(
                    f"Insufficient capital: need ${total_cost:.2f}, "
                    f"have ${account.current_capital:.2f}"
                )

            # Apply slippage
            slippage_factor = account.slippage_bps / 10000
            slippage = total_cost * slippage_factor
            total_cost_with_slippage = total_cost + slippage

            # Create trade record
            trade = SimulationTrade(
                id=str(uuid.uuid4()),
                account_id=account_id,
                opportunity_id=None,
                strategy_type="copy_trading",
                positions_data=[
                    {
                        "market": market_id,
                        "market_question": market_question,
                        "token_id": token_id,
                        "outcome": outcome,
                        "price": price,
                        "size": size,
                    }
                ],
                total_cost=total_cost_with_slippage,
                expected_profit=0.0,  # Unknown for copy trades
                slippage=slippage,
                status=TradeStatus.OPEN,
                copied_from_wallet=copied_from,
            )
            session.add(trade)

            # Create position record with spread trading exits
            # Default: take profit at +5%, stop loss at -10%
            take_profit = min(price * 1.05, 0.99) if price > 0 else None
            stop_loss = max(price * 0.90, 0.01) if price > 0 else None

            position = SimulationPosition(
                id=str(uuid.uuid4()),
                account_id=account_id,
                opportunity_id=None,
                market_id=market_id,
                market_question=market_question,
                token_id=token_id,
                side=PositionSide.YES if outcome.upper() == "YES" else PositionSide.NO,
                quantity=size,
                entry_price=price,
                entry_cost=total_cost_with_slippage,
                take_profit_price=take_profit,
                stop_loss_price=stop_loss,
                status=TradeStatus.OPEN,
            )
            session.add(position)

            # Deduct from capital
            account.current_capital -= total_cost_with_slippage
            account.total_trades += 1

            await session.commit()
            await session.refresh(trade)

            return trade

    async def _close_sim_position(
        self,
        account_id: str,
        market_id: str,
        token_id: str,
        outcome: str,
        sell_price: float,
        copied_from: str,
    ) -> Optional[float]:
        """Close a matching simulation position (mirror a SELL).

        Returns the realized PnL, or None if no matching position found.
        """
        async with AsyncSessionLocal() as session:
            # Find matching open position
            side = PositionSide.YES if outcome.upper() == "YES" else PositionSide.NO

            query = select(SimulationPosition).where(
                and_(
                    SimulationPosition.account_id == account_id,
                    SimulationPosition.market_id == market_id,
                    SimulationPosition.status == TradeStatus.OPEN,
                    SimulationPosition.side == side,
                )
            )
            if token_id:
                query = query.where(SimulationPosition.token_id == token_id)

            result = await session.execute(query)
            position = result.scalar_one_or_none()

            if not position:
                logger.warning(
                    "No matching open position for SELL copy",
                    account_id=account_id,
                    market_id=market_id,
                    outcome=outcome,
                )
                return None

            # Calculate PnL
            sell_value = sell_price * position.quantity
            entry_cost = position.entry_cost
            fee = sell_value * 0.02  # 2% Polymarket fee
            pnl = sell_value - entry_cost - fee

            # Close the position
            position.status = (
                TradeStatus.RESOLVED_WIN if pnl > 0 else TradeStatus.RESOLVED_LOSS
            )
            position.current_price = sell_price
            position.unrealized_pnl = 0.0

            # Update account
            account = await session.get(SimulationAccount, account_id)
            if account:
                account.current_capital += sell_value - fee
                account.total_pnl += pnl
                if pnl > 0:
                    account.winning_trades += 1
                else:
                    account.losing_trades += 1

            # Find and update the associated trade record
            if position.opportunity_id:
                trade_result = await session.execute(
                    select(SimulationTrade).where(
                        and_(
                            SimulationTrade.account_id == account_id,
                            SimulationTrade.opportunity_id == position.opportunity_id,
                            SimulationTrade.status == TradeStatus.OPEN,
                        )
                    )
                )
                sim_trade = trade_result.scalar_one_or_none()
                if sim_trade:
                    sim_trade.status = (
                        TradeStatus.RESOLVED_WIN
                        if pnl > 0
                        else TradeStatus.RESOLVED_LOSS
                    )
                    sim_trade.actual_payout = sell_value - fee
                    sim_trade.actual_pnl = pnl
                    sim_trade.fees_paid = fee
                    sim_trade.resolved_at = utcnow()

            await session.commit()

            return pnl

    # ==================== LIVE TRADING EXECUTION ====================

    async def _execute_live_buy(
        self,
        trade: dict,
        config: CopyTradingConfig,
    ) -> Optional[str]:
        """Execute a copy BUY via the live trading service.

        Returns order ID on success, None on failure.
        """
        from services.trading import trading_service, OrderSide

        if not trading_service.is_ready():
            logger.warning("Live trading not initialized, falling back to simulation")
            return None

        token_id = trade.get("asset", trade.get("assetId", ""))
        source_price = float(trade.get("price", 0) or 0)
        float(trade.get("size", 0) or trade.get("amount", 0) or 0)

        if not token_id or source_price <= 0:
            return None

        # Get current best price
        try:
            current_price = await polymarket_client.get_price(token_id, side="BUY")
        except Exception:
            current_price = source_price

        # Calculate size
        async with AsyncSessionLocal() as session:
            account = await session.get(SimulationAccount, config.account_id)
            capital = account.current_capital if account else 10000.0

        copy_size = self._calculate_copy_size(trade, config, capital)
        if copy_size <= 0:
            return None

        order = await trading_service.place_order(
            token_id=token_id,
            side=OrderSide.BUY,
            price=current_price,
            size=copy_size,
            market_question=trade.get("title", ""),
        )

        if order.status.value == "failed":
            logger.error("Live copy order failed", error=order.error_message)
            return None

        return order.id

    async def _execute_live_sell(
        self,
        trade: dict,
        config: CopyTradingConfig,
    ) -> Optional[str]:
        """Execute a copy SELL via the live trading service."""
        from services.trading import trading_service, OrderSide

        if not trading_service.is_ready():
            return None

        token_id = trade.get("asset", trade.get("assetId", ""))
        source_price = float(trade.get("price", 0) or 0)

        if not token_id or source_price <= 0:
            return None

        try:
            current_price = await polymarket_client.get_price(token_id, side="SELL")
        except Exception:
            current_price = source_price

        source_size = float(trade.get("size", 0) or trade.get("amount", 0) or 0)

        order = await trading_service.place_order(
            token_id=token_id,
            side=OrderSide.SELL,
            price=current_price,
            size=source_size,
        )

        if order.status.value == "failed":
            logger.error("Live sell order failed", error=order.error_message)
            return None

        return order.id

    # ==================== RECORD KEEPING ====================

    async def _record_copied_trade(
        self,
        config: CopyTradingConfig,
        source_trade_id: str,
        market_id: str,
        market_question: str,
        token_id: str,
        side: str,
        outcome: str,
        source_price: float,
        source_size: float,
        source_timestamp: Optional[datetime],
        status: str = "pending",
        executed_price: float = None,
        executed_size: float = None,
        simulation_trade_id: str = None,
        error: str = None,
        realized_pnl: float = None,
    ) -> CopiedTrade:
        """Record a copied trade for deduplication and tracking."""
        async with AsyncSessionLocal() as session:
            copied = CopiedTrade(
                id=str(uuid.uuid4()),
                config_id=config.id,
                source_trade_id=source_trade_id,
                source_wallet=config.source_wallet,
                market_id=market_id,
                market_question=market_question,
                token_id=token_id,
                side=side,
                outcome=outcome,
                source_price=source_price,
                source_size=source_size,
                executed_price=executed_price,
                executed_size=executed_size,
                status=status,
                execution_mode="simulation",
                simulation_trade_id=simulation_trade_id,
                error_message=error,
                source_timestamp=source_timestamp,
                executed_at=utcnow() if status == "executed" else None,
                realized_pnl=realized_pnl,
            )
            session.add(copied)
            await session.commit()
            await session.refresh(copied)
            return copied

    # ==================== MAIN POLL LOOP ====================

    async def _process_config(self, config: CopyTradingConfig):
        """Process a single copy trading config: detect and copy new trades."""
        try:
            new_trades = await self._get_new_trades(config.source_wallet, config)

            if not new_trades:
                return

            logger.info(
                "Found new trades to copy",
                config_id=config.id,
                source_wallet=config.source_wallet,
                count=len(new_trades),
            )

            for trade in new_trades:
                side = (trade.get("side") or "").upper()

                # Check if we should copy this trade
                should_copy, reason = self._should_copy_trade(trade, config)
                if not should_copy:
                    # Record as skipped for audit trail
                    await self._record_copied_trade(
                        config,
                        source_trade_id=trade.get("id", ""),
                        market_id=trade.get("market", trade.get("condition_id", "")),
                        market_question=trade.get("title", ""),
                        token_id=trade.get("asset", ""),
                        side=side,
                        outcome=trade.get("outcome", ""),
                        source_price=float(trade.get("price", 0) or 0),
                        source_size=float(
                            trade.get("size", 0) or trade.get("amount", 0) or 0
                        ),
                        source_timestamp=None,
                        status="skipped",
                        error=reason,
                    )
                    continue

                # In ARB_ONLY mode, check for matching opportunity
                if config.copy_mode == CopyTradingMode.ARB_ONLY:
                    opp = await self._check_arb_match(trade)
                    if not opp:
                        await self._record_copied_trade(
                            config,
                            source_trade_id=trade.get("id", ""),
                            market_id=trade.get(
                                "market", trade.get("condition_id", "")
                            ),
                            market_question=trade.get("title", ""),
                            token_id=trade.get("asset", ""),
                            side=side,
                            outcome=trade.get("outcome", ""),
                            source_price=float(trade.get("price", 0) or 0),
                            source_size=float(
                                trade.get("size", 0) or trade.get("amount", 0) or 0
                            ),
                            source_timestamp=None,
                            status="skipped",
                            error="No matching arbitrage opportunity",
                        )
                        continue

                    if opp.roi_percent < config.min_roi_threshold:
                        await self._record_copied_trade(
                            config,
                            source_trade_id=trade.get("id", ""),
                            market_id=trade.get(
                                "market", trade.get("condition_id", "")
                            ),
                            market_question=trade.get("title", ""),
                            token_id=trade.get("asset", ""),
                            side=side,
                            outcome=trade.get("outcome", ""),
                            source_price=float(trade.get("price", 0) or 0),
                            source_size=float(
                                trade.get("size", 0) or trade.get("amount", 0) or 0
                            ),
                            source_timestamp=None,
                            status="skipped",
                            error=f"ROI {opp.roi_percent:.1f}% below threshold {config.min_roi_threshold}%",
                        )
                        continue

                # Execute the copy
                if side == "BUY":
                    await self._execute_copy_buy(trade, config)
                elif side == "SELL":
                    await self._execute_copy_sell(trade, config)

        except Exception as e:
            logger.error(
                "Error processing copy config",
                config_id=config.id,
                error=str(e),
            )

    async def _poll_loop(self):
        """Main polling loop for copy trading"""
        while self._running:
            if not global_pause_state.is_paused:
                try:
                    # Get all enabled configs from DB (fresh read each cycle)
                    async with AsyncSessionLocal() as session:
                        result = await session.execute(
                            select(CopyTradingConfig).where(CopyTradingConfig.enabled)
                        )
                        configs = list(result.scalars().all())

                    # Process each config concurrently
                    if configs:
                        tasks = [self._process_config(config) for config in configs]
                        await asyncio.gather(*tasks, return_exceptions=True)

                except Exception as e:
                    logger.error("Copy trading poll error", error=str(e))

            await asyncio.sleep(self._poll_interval)

    # ==================== SERVICE LIFECYCLE ====================

    async def _on_realtime_trade(self, event):
        """Callback for real-time WebSocket wallet trade events.

        Uses DIRECT event processing (no HTTP API re-fetch) for <500ms latency.
        The WalletTradeEvent already contains all trade data from on-chain parsing.
        """
        if global_pause_state.is_paused:
            return
        try:
            wallet_address = event.wallet_address.lower()
            # Find configs that track this wallet
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(CopyTradingConfig).where(
                        CopyTradingConfig.enabled,
                        CopyTradingConfig.source_wallet == wallet_address,
                    )
                )
                configs = list(result.scalars().all())

            if configs:
                logger.info(
                    "Real-time trade detected — direct processing",
                    wallet=wallet_address,
                    configs=len(configs),
                    side=event.side,
                    size=event.size,
                    price=event.price,
                    token_id=event.token_id,
                    detection_latency_ms=round(event.latency_ms, 1),
                    tx_hash=event.tx_hash,
                )

                # Broadcast detection event to frontend immediately
                await self._broadcast_copy_event(
                    "copy_trade_detected",
                    {
                        "wallet": wallet_address,
                        "side": event.side,
                        "token_id": event.token_id,
                        "size": event.size,
                        "price": event.price,
                        "tx_hash": event.tx_hash,
                        "detection_latency_ms": round(event.latency_ms, 1),
                        "configs_count": len(configs),
                    },
                )

                # Process directly from the event data — no HTTP API call
                tasks = [
                    self._process_realtime_event(event, config)
                    for config in configs
                ]
                await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.error("Error handling real-time trade event", error=str(e))

    async def start(self):
        """Start copy trading service"""
        if self._running:
            return

        self._running = True
        logger.info("Starting copy trading service")

        # Load existing configs
        configs = await self.get_configs()
        for config in configs:
            self._active_configs[config.id] = config

        # Start WebSocket monitor for real-time trade detection
        try:
            from services.wallet_ws_monitor import wallet_ws_monitor

            wallet_ws_monitor.set_wallets_for_source(
                "copy_trader",
                [config.source_wallet for config in configs if config.enabled],
            )
            wallet_ws_monitor.add_callback(self._on_realtime_trade)
            asyncio.create_task(wallet_ws_monitor.start())
            logger.info("WebSocket wallet monitor started for copy trading")
        except Exception as e:
            logger.warning(f"WebSocket monitor unavailable, using polling only: {e}")

        # Polling loop as fallback / supplement
        asyncio.create_task(self._poll_loop())

    def stop(self):
        """Stop copy trading service"""
        self._running = False
        logger.info("Stopped copy trading service")

    # ==================== STATS & QUERIES ====================

    async def get_copy_stats(self, config_id: str) -> Optional[dict]:
        """Get statistics for a copy trading configuration"""
        async with AsyncSessionLocal() as session:
            config = await session.get(CopyTradingConfig, config_id)
            if not config:
                return None

            success_rate = (
                config.successful_copies / config.total_copied * 100
                if config.total_copied > 0
                else 0
            )

            return {
                "config_id": config.id,
                "source_wallet": config.source_wallet,
                "account_id": config.account_id,
                "enabled": config.enabled,
                "copy_mode": config.copy_mode.value,
                "total_copied": config.total_copied,
                "successful_copies": config.successful_copies,
                "failed_copies": config.failed_copies,
                "success_rate": success_rate,
                "total_pnl": config.total_pnl,
                "total_buys_copied": config.total_buys_copied,
                "total_sells_copied": config.total_sells_copied,
                "settings": {
                    "min_roi_threshold": config.min_roi_threshold,
                    "max_position_size": config.max_position_size,
                    "copy_delay_seconds": config.copy_delay_seconds,
                    "slippage_tolerance": config.slippage_tolerance,
                    "proportional_sizing": config.proportional_sizing,
                    "proportional_multiplier": config.proportional_multiplier,
                    "copy_buys": config.copy_buys,
                    "copy_sells": config.copy_sells,
                    "market_categories": config.market_categories,
                },
            }

    async def get_copied_trades(
        self,
        config_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """Get history of copied trades with optional filters."""
        async with AsyncSessionLocal() as session:
            query = select(CopiedTrade).order_by(CopiedTrade.copied_at.desc())

            if config_id:
                query = query.where(CopiedTrade.config_id == config_id)
            if status:
                query = query.where(CopiedTrade.status == status)

            query = query.offset(offset).limit(limit)
            result = await session.execute(query)
            trades = result.scalars().all()

            return [
                {
                    "id": t.id,
                    "config_id": t.config_id,
                    "source_trade_id": t.source_trade_id,
                    "source_wallet": t.source_wallet,
                    "market_id": t.market_id,
                    "market_question": t.market_question,
                    "token_id": t.token_id,
                    "side": t.side,
                    "outcome": t.outcome,
                    "source_price": t.source_price,
                    "source_size": t.source_size,
                    "executed_price": t.executed_price,
                    "executed_size": t.executed_size,
                    "status": t.status,
                    "execution_mode": t.execution_mode,
                    "error_message": t.error_message,
                    "source_timestamp": t.source_timestamp.isoformat()
                    if t.source_timestamp
                    else None,
                    "copied_at": t.copied_at.isoformat() if t.copied_at else None,
                    "executed_at": t.executed_at.isoformat() if t.executed_at else None,
                    "realized_pnl": t.realized_pnl,
                }
                for t in trades
            ]

    async def get_source_wallet_positions(self, wallet_address: str) -> list[dict]:
        """Get current positions for a source wallet."""
        try:
            return await polymarket_client.get_wallet_positions_with_prices(
                wallet_address
            )
        except Exception as e:
            logger.error(
                "Error fetching source wallet positions",
                wallet=wallet_address,
                error=str(e),
            )
            return []

    async def force_sync(self, config_id: str) -> dict:
        """Force an immediate sync for a specific config.

        Useful for catching up after downtime or initial setup.
        """
        async with AsyncSessionLocal() as session:
            config = await session.get(CopyTradingConfig, config_id)
            if not config:
                raise ValueError(f"Config not found: {config_id}")

        await self._process_config(config)

        return {"message": "Sync complete", "config_id": config_id}


# Singleton instance
copy_trader = CopyTradingService()
