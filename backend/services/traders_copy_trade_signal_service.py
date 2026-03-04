from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import and_, or_, select

from models.database import AsyncSessionLocal, DiscoveredWallet, Trader, TraderGroupMember, TrackedWallet
from services.live_execution_service import live_execution_service
from services.polymarket import polymarket_client
from services.signal_bus import make_dedupe_key
from services.strategy_loader import strategy_loader
from services.strategy_signal_bridge import bridge_opportunities_to_signals
from services.strategy_sdk import StrategySDK
from services.strategies.traders_copy_trade import validate_traders_copy_trade_config
from services.wallet_ws_monitor import WalletMonitorEvent, WalletTradeEvent, wallet_ws_monitor
from utils.converters import safe_float
from utils.logger import get_logger
from utils.utcnow import utcnow

logger = get_logger("traders_copy_trade_signal_service")


@dataclass
class _TokenMarketSnapshot:
    market_id: str
    market_question: str
    market_slug: str | None
    outcome: str
    liquidity: float | None


def _wallet_position_token_id(position: dict[str, Any]) -> str:
    for key in ("asset", "asset_id", "assetId", "token_id", "tokenId", "token"):
        token_id = str(position.get(key) or "").strip()
        if token_id:
            return token_id
    return ""


def _wallet_position_price(position: dict[str, Any], size: float) -> float:
    for key in ("avgPrice", "avg_price", "price", "curPrice", "currentPrice"):
        value = safe_float(position.get(key), 0.0)
        if value > 0.0:
            return value
    initial_value = safe_float(position.get("initialValue"), safe_float(position.get("initial_value"), 0.0))
    if initial_value > 0.0 and size > 0.0:
        return initial_value / size
    current_value = safe_float(position.get("currentValue"), safe_float(position.get("current_value"), 0.0))
    if current_value > 0.0 and size > 0.0:
        return current_value / size
    return 0.0


class TradersCopyTradeSignalService:
    def __init__(self) -> None:
        self._running = False
        self._sync_task: asyncio.Task | None = None
        self._processor_task: asyncio.Task | None = None
        self._replay_task: asyncio.Task | None = None
        self._queue: asyncio.Queue[WalletTradeEvent] = asyncio.Queue(maxsize=5000)
        self._callback_registered = False
        self._tracked_wallets: set[str] = set()
        self._recent_events: dict[str, datetime] = {}
        self._token_cache: dict[str, tuple[datetime, _TokenMarketSnapshot]] = {}
        self._execution_wallet: str = ""
        self._scope_refresh_interval_seconds = 15
        self._event_dedupe_ttl_seconds = 900
        self._token_cache_ttl_seconds = 300
        self._replay_interval_seconds = 2.0
        self._replay_batch_size = 500
        self._replay_cursor_detected_at: datetime | None = None
        self._replay_cursor_id: str = ""
        self._replay_wakeup = asyncio.Event()
        self._overflow_direct_process_semaphore = asyncio.Semaphore(16)
        self._strategy_missing_warned = False
        self._background_tasks: set[asyncio.Task] = set()

    async def start(self) -> None:
        if self._running:
            return
        self._running = True

        if not self._callback_registered:
            wallet_ws_monitor.add_callback(self._on_wallet_trade)
            self._callback_registered = True

        await self._refresh_tracked_wallets()
        await wallet_ws_monitor.start()

        self._replay_cursor_detected_at = utcnow()
        self._replay_cursor_id = ""
        self._sync_task = asyncio.create_task(self._scope_sync_loop())
        self._processor_task = asyncio.create_task(self._processor_loop())
        self._replay_task = asyncio.create_task(self._replay_loop())
        logger.info("Traders copy-trade signal service started")

    async def refresh_scope(self) -> None:
        await self._refresh_tracked_wallets()
        self._replay_wakeup.set()

    def stop(self) -> None:
        self._running = False
        if self._sync_task and not self._sync_task.done():
            self._sync_task.cancel()
        if self._processor_task and not self._processor_task.done():
            self._processor_task.cancel()
        if self._replay_task and not self._replay_task.done():
            self._replay_task.cancel()
        for task in list(self._background_tasks):
            if not task.done():
                task.cancel()
        self._replay_wakeup.set()

    async def _scope_sync_loop(self) -> None:
        while self._running:
            try:
                await self._refresh_tracked_wallets()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Copy-trade scope refresh failed", exc_info=exc)
            await asyncio.sleep(self._scope_refresh_interval_seconds)

    async def _processor_loop(self) -> None:
        while self._running:
            try:
                event = await self._queue.get()
            except asyncio.CancelledError:
                raise
            try:
                await self._process_wallet_trade_event(event)
            except Exception as exc:
                logger.warning("Copy-trade signal processing failed", exc_info=exc)
            finally:
                self._queue.task_done()

    async def _replay_loop(self) -> None:
        while self._running:
            self._replay_wakeup.clear()
            try:
                await self._replay_wallet_monitor_events(max_batches=10)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Copy-trade replay processing failed", exc_info=exc)
            try:
                await asyncio.wait_for(self._replay_wakeup.wait(), timeout=self._replay_interval_seconds)
            except asyncio.TimeoutError:
                pass

    async def _process_overflow_event_direct(self, event: WalletTradeEvent) -> None:
        async with self._overflow_direct_process_semaphore:
            try:
                await self._process_wallet_trade_event(event)
            except Exception as exc:
                logger.warning("Copy-trade overflow fallback processing failed", exc_info=exc)

    def _spawn_background_task(self, coro: Any) -> None:
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _replay_wallet_monitor_events(self, *, max_batches: int = 1) -> int:
        if not self._tracked_wallets:
            return 0
        cursor_detected_at = self._replay_cursor_detected_at or utcnow()
        cursor_id = str(self._replay_cursor_id or "")
        processed = 0
        tracked_wallets = sorted(self._tracked_wallets)
        for _ in range(max_batches):
            async with AsyncSessionLocal() as session:
                rows = (
                    (
                        await session.execute(
                            select(WalletMonitorEvent)
                            .where(WalletMonitorEvent.wallet_address.in_(tracked_wallets))
                            .where(
                                or_(
                                    WalletMonitorEvent.detected_at > cursor_detected_at,
                                    and_(
                                        WalletMonitorEvent.detected_at == cursor_detected_at,
                                        WalletMonitorEvent.id > cursor_id,
                                    ),
                                )
                            )
                            .order_by(WalletMonitorEvent.detected_at.asc(), WalletMonitorEvent.id.asc())
                            .limit(self._replay_batch_size)
                        )
                    )
                    .scalars()
                    .all()
                )
            if not rows:
                break
            for row in rows:
                detected_at = row.detected_at if isinstance(row.detected_at, datetime) else utcnow()
                replay_event = WalletTradeEvent(
                    wallet_address=str(row.wallet_address or ""),
                    token_id=str(row.token_id or ""),
                    side=str(row.side or ""),
                    size=max(0.0, safe_float(row.size, 0.0)),
                    price=max(0.0, safe_float(row.price, 0.0)),
                    tx_hash=str(row.tx_hash or ""),
                    order_hash=str(row.order_hash or ""),
                    log_index=int(row.log_index or 0),
                    block_number=int(row.block_number or 0),
                    timestamp=detected_at,
                    detected_at=detected_at,
                    latency_ms=max(0.0, safe_float(row.detection_latency_ms, 0.0)),
                )
                await self._process_wallet_trade_event(replay_event)
                processed += 1
                cursor_detected_at = detected_at
                cursor_id = str(row.id or "")
            self._replay_cursor_detected_at = cursor_detected_at
            self._replay_cursor_id = cursor_id
            if len(rows) < self._replay_batch_size:
                break
        return processed

    async def _on_wallet_trade(self, event: WalletTradeEvent) -> None:
        if not self._running:
            return
        wallet = str(event.wallet_address or "").strip().lower()
        if not wallet or wallet not in self._tracked_wallets:
            return
        if self._execution_wallet and wallet == self._execution_wallet:
            return
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.warning(
                "Copy-trade signal queue full; replay fallback activated",
                wallet=wallet,
                tx_hash=str(event.tx_hash or ""),
                log_index=int(event.log_index or 0),
            )
            self._replay_wakeup.set()
            self._spawn_background_task(self._process_overflow_event_direct(event))

    async def _refresh_tracked_wallets(self) -> None:
        scopes: list[dict[str, Any]] = []
        async with AsyncSessionLocal() as session:
            traders = (
                (
                    await session.execute(
                        select(Trader.source_configs_json).where(
                            Trader.is_enabled == True,  # noqa: E712
                            Trader.is_paused == False,  # noqa: E712
                        )
                    )
                )
                .scalars()
                .all()
            )

            for source_configs in traders:
                if not isinstance(source_configs, list):
                    continue
                for source_config in source_configs:
                    if not isinstance(source_config, dict):
                        continue
                    if str(source_config.get("source_key") or "").strip().lower() != "traders":
                        continue
                    if str(source_config.get("strategy_key") or "").strip().lower() != "traders_copy_trade":
                        continue
                    params = validate_traders_copy_trade_config(source_config.get("strategy_params") or {})
                    scopes.append(StrategySDK.validate_trader_scope_config(params.get("traders_scope")))

            combined = await self._resolve_wallets_for_scopes(session, scopes)

        execution_wallet = str(live_execution_service.get_execution_wallet_address() or "").strip().lower()
        if execution_wallet:
            combined.discard(execution_wallet)
        self._execution_wallet = execution_wallet
        self._tracked_wallets = combined
        wallet_ws_monitor.set_wallets_for_source("traders_copy_trade", sorted(combined))

    async def _resolve_wallets_for_scopes(self, session: Any, scopes: list[dict[str, Any]]) -> set[str]:
        tracked_needed = any("tracked" in set(scope.get("modes") or []) for scope in scopes)
        pool_needed = any("pool" in set(scope.get("modes") or []) for scope in scopes)
        group_ids: set[str] = set()
        individual_wallets: set[str] = set()

        for scope in scopes:
            modes = {str(mode or "").strip().lower() for mode in (scope.get("modes") or [])}
            if "group" in modes:
                for raw_group_id in scope.get("group_ids") or []:
                    group_id = str(raw_group_id or "").strip()
                    if group_id:
                        group_ids.add(group_id)
            if "individual" in modes:
                for raw_wallet in scope.get("individual_wallets") or []:
                    wallet = StrategySDK.normalize_trader_wallet(raw_wallet)
                    if wallet:
                        individual_wallets.add(wallet)

        tracked_wallets: set[str] = set()
        pool_wallets: set[str] = set()
        group_wallets: set[str] = set()

        if tracked_needed:
            tracked_rows = (await session.execute(select(TrackedWallet.address))).scalars().all()
            tracked_wallets = {StrategySDK.normalize_trader_wallet(address) for address in tracked_rows if address}
            tracked_wallets.discard("")

        if pool_needed:
            pool_rows = (
                (
                    await session.execute(
                        select(DiscoveredWallet.address).where(DiscoveredWallet.in_top_pool == True)  # noqa: E712
                    )
                )
                .scalars()
                .all()
            )
            pool_wallets = {StrategySDK.normalize_trader_wallet(address) for address in pool_rows if address}
            pool_wallets.discard("")

        if group_ids:
            group_rows = (
                (
                    await session.execute(
                        select(TraderGroupMember.wallet_address).where(TraderGroupMember.group_id.in_(list(group_ids)))
                    )
                )
                .scalars()
                .all()
            )
            group_wallets = {StrategySDK.normalize_trader_wallet(address) for address in group_rows if address}
            group_wallets.discard("")

        return tracked_wallets | pool_wallets | group_wallets | individual_wallets

    def _is_duplicate_event(self, event: WalletTradeEvent) -> bool:
        now = utcnow()
        event_key = make_dedupe_key(
            "traders_copy_trade",
            str(event.wallet_address or "").strip().lower(),
            str(event.tx_hash or "").strip().lower(),
            str(event.order_hash or "").strip().lower(),
            int(event.log_index or 0),
            str(event.token_id or "").strip(),
            str(event.side or "").strip().upper(),
            int(event.block_number or 0),
            round(max(0.0, safe_float(event.price, 0.0)), 8),
            round(max(0.0, safe_float(event.size, 0.0)), 8),
        )
        stale_before = now - timedelta(seconds=self._event_dedupe_ttl_seconds)
        stale_keys = [key for key, seen_at in self._recent_events.items() if seen_at < stale_before]
        for key in stale_keys:
            self._recent_events.pop(key, None)
        if event_key in self._recent_events:
            return True
        self._recent_events[event_key] = now
        return False

    def _resolve_copy_trade_strategy(self) -> Any | None:
        runtime_strategy = strategy_loader.get_instance("traders_copy_trade")
        if runtime_strategy is not None and callable(getattr(runtime_strategy, "detect_async", None)):
            self._strategy_missing_warned = False
            return runtime_strategy
        if not self._strategy_missing_warned:
            logger.warning("Copy-trade runtime strategy unavailable", strategy_key="traders_copy_trade")
            self._strategy_missing_warned = True
        return None

    async def _process_wallet_trade_event(self, event: WalletTradeEvent, *, strategy: Any | None = None) -> int:
        if self._is_duplicate_event(event):
            return 0

        side = str(event.side or "").strip().upper()
        if side not in {"BUY", "SELL"}:
            return 0

        token_id = str(event.token_id or "").strip()
        if not token_id:
            return 0

        source_wallet = str(event.wallet_address or "").strip().lower()

        entry_price = safe_float(event.price, 0.0)
        size = max(0.0, safe_float(event.size, 0.0))
        if entry_price <= 0.0 or size <= 0.0:
            return 0

        market = await self._resolve_market_snapshot(token_id)
        if market.outcome not in {"YES", "NO"}:
            logger.warning(
                "Skipping copy-trade signal; unresolved token outcome",
                wallet=source_wallet,
                token_id=token_id,
                tx_hash=str(event.tx_hash or ""),
                block_number=int(event.block_number or 0),
            )
            return 0

        source_item_id = (
            f"{event.tx_hash}:{source_wallet}:{token_id}:{side}:{int(event.log_index or 0)}:{str(event.order_hash or '')}"
        )
        dedupe_key = make_dedupe_key(
            "traders_copy_trade_signal",
            source_wallet,
            str(event.tx_hash or "").strip().lower(),
            str(event.order_hash or "").strip().lower(),
            int(event.log_index or 0),
            token_id,
            side,
            int(event.block_number or 0),
            round(entry_price, 8),
            round(size, 8),
        )

        source_notional = entry_price * size
        copy_event_payload = {
            "wallet_address": source_wallet,
            "token_id": token_id,
            "side": side,
            "size": size,
            "price": entry_price,
            "tx_hash": str(event.tx_hash or ""),
            "order_hash": str(event.order_hash or ""),
            "log_index": int(event.log_index or 0),
            "block_number": int(event.block_number or 0),
            "timestamp": event.timestamp.isoformat() if isinstance(event.timestamp, datetime) else None,
            "detected_at": event.detected_at.isoformat() if isinstance(event.detected_at, datetime) else None,
            "latency_ms": max(0.0, safe_float(event.latency_ms, 0.0)),
            "confidence": 0.70,
        }
        source_trade_payload = {
            "wallet_address": source_wallet,
            "side": side,
            "source_notional_usd": source_notional,
            "size": size,
            "price": entry_price,
            "tx_hash": str(event.tx_hash or ""),
            "order_hash": str(event.order_hash or ""),
            "log_index": int(event.log_index or 0),
            "detected_at": event.detected_at.isoformat() if isinstance(event.detected_at, datetime) else None,
        }
        market_payload = {
            "market_id": market.market_id,
            "market_question": market.market_question,
            "market_slug": market.market_slug,
            "outcome": market.outcome,
            "liquidity": market.liquidity,
            "token_id": token_id,
        }
        strategy_input = [
            {
                "copy_event": copy_event_payload,
                "source_trade": source_trade_payload,
                "market": market_payload,
                "source_item_id": source_item_id,
                "dedupe_key": dedupe_key,
            }
        ]

        runtime_strategy = strategy
        if runtime_strategy is None:
            runtime_strategy = self._resolve_copy_trade_strategy()
        if runtime_strategy is None:
            return 0
        opportunities = await runtime_strategy.detect_async(strategy_input, [], {})
        if not opportunities:
            return 0

        async with AsyncSessionLocal() as session:
            await bridge_opportunities_to_signals(
                session,
                opportunities,
                source="traders",
                signal_type_override="copy_trade",
                default_ttl_minutes=15,
                sweep_missing=False,
                refresh_prices=False,
            )
        return len(opportunities)

    async def copy_existing_open_positions_for_trader(
        self,
        *,
        trader_id: str,
        copy_existing_positions: bool | None = None,
    ) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "trader_id": trader_id,
            "status": "skipped",
            "reason": "",
            "wallets_targeted": 0,
            "wallets_scanned": 0,
            "positions_seen": 0,
            "positions_queued": 0,
            "signals_created": 0,
            "errors": [],
        }
        if copy_existing_positions is False:
            summary["reason"] = "operator_disabled_copy_existing_positions"
            return summary

        scopes: list[dict[str, Any]] = []
        async with AsyncSessionLocal() as session:
            trader = await session.get(Trader, trader_id)
            if trader is None:
                summary["status"] = "not_found"
                summary["reason"] = "trader_not_found"
                return summary

            source_configs = trader.source_configs_json if isinstance(trader.source_configs_json, list) else []
            for source_config in source_configs:
                if not isinstance(source_config, dict):
                    continue
                if str(source_config.get("source_key") or "").strip().lower() != "traders":
                    continue
                if str(source_config.get("strategy_key") or "").strip().lower() != "traders_copy_trade":
                    continue
                params = validate_traders_copy_trade_config(source_config.get("strategy_params") or {})
                should_copy_existing = (
                    bool(copy_existing_positions)
                    if copy_existing_positions is not None
                    else bool(params.get("copy_existing_positions_on_start", False))
                )
                if not should_copy_existing:
                    continue
                scopes.append(StrategySDK.validate_trader_scope_config(params.get("traders_scope")))

            if not scopes:
                summary["reason"] = "copy_existing_positions_not_enabled"
                return summary

            wallets = await self._resolve_wallets_for_scopes(session, scopes)

        execution_wallet = str(live_execution_service.get_execution_wallet_address() or "").strip().lower()
        if execution_wallet:
            wallets.discard(execution_wallet)
        self._execution_wallet = execution_wallet

        summary["wallets_targeted"] = len(wallets)
        if not wallets:
            summary["reason"] = "no_wallets_in_scope"
            return summary

        strategy = self._resolve_copy_trade_strategy()
        if strategy is None:
            summary["status"] = "failed"
            summary["reason"] = "copy_strategy_unavailable"
            return summary

        now = utcnow()
        seen_wallet_tokens: set[str] = set()
        for wallet in sorted(wallets):
            summary["wallets_scanned"] = int(summary["wallets_scanned"]) + 1
            try:
                wallet_positions = await polymarket_client.get_wallet_positions_with_prices(wallet)
            except Exception as exc:
                errors = list(summary.get("errors") or [])
                errors.append({"wallet": wallet, "error": str(exc)})
                summary["errors"] = errors
                continue
            if not isinstance(wallet_positions, list):
                continue
            for position in wallet_positions:
                if not isinstance(position, dict):
                    continue
                summary["positions_seen"] = int(summary["positions_seen"]) + 1
                token_id = _wallet_position_token_id(position)
                if not token_id:
                    continue
                dedupe_token = f"{wallet}:{token_id}".lower()
                if dedupe_token in seen_wallet_tokens:
                    continue

                size = max(0.0, safe_float(position.get("size"), 0.0))
                if size <= 0.0:
                    continue
                price = _wallet_position_price(position, size)
                if price <= 0.0:
                    continue

                seen_wallet_tokens.add(dedupe_token)
                bootstrap_event = WalletTradeEvent(
                    wallet_address=wallet,
                    token_id=token_id,
                    side="BUY",
                    size=size,
                    price=price,
                    tx_hash=f"bootstrap:{trader_id}:{wallet}:{token_id}",
                    order_hash="bootstrap_open_position",
                    log_index=0,
                    block_number=0,
                    timestamp=now,
                    detected_at=now,
                    latency_ms=0.0,
                )
                summary["positions_queued"] = int(summary["positions_queued"]) + 1
                created = await self._process_wallet_trade_event(bootstrap_event, strategy=strategy)
                summary["signals_created"] = int(summary["signals_created"]) + int(created)

        summary["status"] = "completed"
        summary["reason"] = "ok"
        return summary

    async def _resolve_market_snapshot(self, token_id: str) -> _TokenMarketSnapshot:
        now = utcnow()
        cached = self._token_cache.get(token_id)
        if cached is not None:
            cached_at, payload = cached
            if (now - cached_at).total_seconds() <= self._token_cache_ttl_seconds:
                return payload

        market_id = f"token:{token_id}"
        market_question = f"Token {token_id}"
        market_slug: str | None = None
        outcome = ""
        liquidity: float | None = None

        try:
            info = await polymarket_client.get_market_by_token_id(token_id)
        except Exception:
            if cached is not None:
                return cached[1]
            info = None

        if isinstance(info, dict):
            condition_id = str(info.get("condition_id") or info.get("id") or "").strip()
            question = str(info.get("question") or "").strip()
            slug = str(info.get("event_slug") or info.get("slug") or "").strip() or None
            market_id = condition_id or market_id
            market_question = question or market_question
            market_slug = slug
            liquidity = safe_float(info.get("liquidity"), None)

            tokens = info.get("tokens") if isinstance(info.get("tokens"), list) else []
            for raw_token in tokens:
                if not isinstance(raw_token, dict):
                    continue
                raw_token_id = str(raw_token.get("token_id") or raw_token.get("id") or "").strip()
                if raw_token_id != token_id:
                    continue
                raw_outcome = str(raw_token.get("outcome") or "").strip().upper()
                if raw_outcome in {"YES", "NO"}:
                    outcome = raw_outcome
                break

        snapshot = _TokenMarketSnapshot(
            market_id=market_id,
            market_question=market_question,
            market_slug=market_slug,
            outcome=outcome,
            liquidity=liquidity,
        )
        if outcome in {"YES", "NO"}:
            self._token_cache[token_id] = (now, snapshot)
        return snapshot


traders_copy_trade_signal_service = TradersCopyTradeSignalService()
