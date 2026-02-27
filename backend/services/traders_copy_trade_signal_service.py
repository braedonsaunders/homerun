from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import select

from models.database import AsyncSessionLocal, DiscoveredWallet, Trader, TraderGroupMember, TrackedWallet
from services.polymarket import polymarket_client
from services.signal_bus import make_dedupe_key, upsert_trade_signal
from services.strategy_sdk import StrategySDK
from services.wallet_ws_monitor import WalletTradeEvent, wallet_ws_monitor
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


class TradersCopyTradeSignalService:
    def __init__(self) -> None:
        self._running = False
        self._sync_task: asyncio.Task | None = None
        self._processor_task: asyncio.Task | None = None
        self._queue: asyncio.Queue[WalletTradeEvent] = asyncio.Queue(maxsize=5000)
        self._callback_registered = False
        self._tracked_wallets: set[str] = set()
        self._recent_events: dict[str, datetime] = {}
        self._token_cache: dict[str, tuple[datetime, _TokenMarketSnapshot]] = {}
        self._scope_refresh_interval_seconds = 15
        self._event_dedupe_ttl_seconds = 900
        self._token_cache_ttl_seconds = 300

    async def start(self) -> None:
        if self._running:
            return
        self._running = True

        if not self._callback_registered:
            wallet_ws_monitor.add_callback(self._on_wallet_trade)
            self._callback_registered = True

        await self._refresh_tracked_wallets()
        await wallet_ws_monitor.start()

        self._sync_task = asyncio.create_task(self._scope_sync_loop())
        self._processor_task = asyncio.create_task(self._processor_loop())
        logger.info("Traders copy-trade signal service started")

    def stop(self) -> None:
        self._running = False
        if self._sync_task and not self._sync_task.done():
            self._sync_task.cancel()
        if self._processor_task and not self._processor_task.done():
            self._processor_task.cancel()

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

    async def _on_wallet_trade(self, event: WalletTradeEvent) -> None:
        if not self._running:
            return
        wallet = str(event.wallet_address or "").strip().lower()
        if not wallet or wallet not in self._tracked_wallets:
            return
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.warning("Copy-trade signal queue full; dropping event")

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
                    params = StrategySDK.validate_traders_copy_trade_config(source_config.get("strategy_params") or {})
                    scopes.append(StrategySDK.validate_trader_scope_config(params.get("traders_scope")))

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
                        wallet = str(raw_wallet or "").strip().lower()
                        if wallet:
                            individual_wallets.add(wallet)

            tracked_wallets: set[str] = set()
            pool_wallets: set[str] = set()
            group_wallets: set[str] = set()

            if tracked_needed:
                tracked_rows = (await session.execute(select(TrackedWallet.address))).scalars().all()
                tracked_wallets = {str(address or "").strip().lower() for address in tracked_rows if address}

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
                pool_wallets = {str(address or "").strip().lower() for address in pool_rows if address}

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
                group_wallets = {str(address or "").strip().lower() for address in group_rows if address}

        combined = tracked_wallets | pool_wallets | group_wallets | individual_wallets
        self._tracked_wallets = combined
        wallet_ws_monitor.set_wallets_for_source("traders_copy_trade", sorted(combined))

    def _is_duplicate_event(self, event: WalletTradeEvent) -> bool:
        now = utcnow()
        event_key = make_dedupe_key(
            "traders_copy_trade",
            str(event.wallet_address or "").strip().lower(),
            str(event.tx_hash or "").strip().lower(),
            str(event.token_id or "").strip(),
            str(event.side or "").strip().upper(),
            int(event.block_number or 0),
        )
        stale_before = now - timedelta(seconds=self._event_dedupe_ttl_seconds)
        stale_keys = [key for key, seen_at in self._recent_events.items() if seen_at < stale_before]
        for key in stale_keys:
            self._recent_events.pop(key, None)
        if event_key in self._recent_events:
            return True
        self._recent_events[event_key] = now
        return False

    async def _process_wallet_trade_event(self, event: WalletTradeEvent) -> None:
        if self._is_duplicate_event(event):
            return

        side = str(event.side or "").strip().upper()
        if side not in {"BUY", "SELL"}:
            return

        token_id = str(event.token_id or "").strip()
        if not token_id:
            return

        entry_price = safe_float(event.price, 0.0)
        size = max(0.0, safe_float(event.size, 0.0))
        if entry_price <= 0.0 or size <= 0.0:
            return

        market = await self._resolve_market_snapshot(token_id)
        direction = "buy_yes" if market.outcome == "YES" else "buy_no"
        signal_type = "single_wallet_buy" if side == "BUY" else "single_wallet_sell"
        source_notional = entry_price * size
        confidence = 0.70
        edge_percent = max(0.0, abs(0.5 - entry_price) * 200.0) if 0.0 <= entry_price <= 1.0 else 0.0

        source_wallet = str(event.wallet_address or "").strip().lower()
        source_item_id = f"{event.tx_hash}:{source_wallet}:{token_id}:{side}"
        dedupe_key = make_dedupe_key(
            "traders_copy_trade_signal",
            source_wallet,
            str(event.tx_hash or "").strip().lower(),
            token_id,
            side,
            int(event.block_number or 0),
        )

        strategy_context = {
            "source_key": "traders",
            "strategy_slug": "traders_copy_trade",
            "traders_channel": "copy_trade",
            "copy_event": {
                "wallet_address": source_wallet,
                "token_id": token_id,
                "side": side,
                "size": size,
                "price": entry_price,
                "tx_hash": str(event.tx_hash or ""),
                "block_number": int(event.block_number or 0),
                "timestamp": event.timestamp.isoformat() if isinstance(event.timestamp, datetime) else None,
                "detected_at": event.detected_at.isoformat() if isinstance(event.detected_at, datetime) else None,
                "latency_ms": max(0.0, safe_float(event.latency_ms, 0.0)),
                "confidence": confidence,
                "market_id": market.market_id,
                "market_question": market.market_question,
                "market_slug": market.market_slug,
                "outcome": market.outcome,
            },
        }

        payload_json = {
            "wallets": [source_wallet],
            "wallet_addresses": [source_wallet],
            "token_id": token_id,
            "selected_token_id": token_id,
            "market_id": market.market_id,
            "market_question": market.market_question,
            "market_slug": market.market_slug,
            "source_trade": {
                "wallet_address": source_wallet,
                "side": side,
                "source_notional_usd": source_notional,
                "size": size,
                "price": entry_price,
                "tx_hash": str(event.tx_hash or ""),
                "detected_at": event.detected_at.isoformat() if isinstance(event.detected_at, datetime) else None,
            },
            "strategy_context": strategy_context,
            "execution_plan": {
                "policy": "REPRICE_LOOP",
                "time_in_force": "IOC",
                "legs": [
                    {
                        "leg_id": "copy_leg_1",
                        "market_id": market.market_id,
                        "market_question": market.market_question,
                        "token_id": token_id,
                        "side": "buy" if side == "BUY" else "sell",
                        "outcome": market.outcome,
                        "limit_price": entry_price,
                        "price_policy": "taker_limit",
                        "time_in_force": "IOC",
                        "notional_weight": 1.0,
                        "min_fill_ratio": 0.0,
                        "metadata": {
                            "source_wallet": source_wallet,
                            "copy_trade": True,
                            "source_tx_hash": str(event.tx_hash or ""),
                        },
                    }
                ],
            },
        }

        expires_at = utcnow() + timedelta(seconds=900)

        async with AsyncSessionLocal() as session:
            await upsert_trade_signal(
                session,
                source="traders",
                source_item_id=source_item_id,
                signal_type=signal_type,
                strategy_type="traders_copy_trade",
                market_id=market.market_id,
                market_question=market.market_question,
                direction=direction,
                entry_price=entry_price,
                edge_percent=edge_percent,
                confidence=confidence,
                liquidity=market.liquidity,
                expires_at=expires_at,
                payload_json=payload_json,
                strategy_context_json=strategy_context,
                dedupe_key=dedupe_key,
                commit=True,
            )

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
        outcome = "NO"
        liquidity: float | None = None

        try:
            info = await polymarket_client.get_market_by_token_id(token_id)
        except Exception:
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
        self._token_cache[token_id] = (now, snapshot)
        return snapshot


traders_copy_trade_signal_service = TradersCopyTradeSignalService()
