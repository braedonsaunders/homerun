import asyncio
from utils.utcnow import utcnow
from typing import Optional
from sqlalchemy import select

from services.polymarket import polymarket_client
from services.pause_state import global_pause_state
from models.database import TrackedWallet, AsyncSessionLocal


class WalletTracker:
    """Track specific wallets for trade activity"""

    def __init__(self):
        self.client = polymarket_client
        self.tracked_wallets: dict[
            str, dict
        ] = {}  # In-memory cache with positions/trades
        self._running = False
        self._callbacks: list[callable] = []
        self._initialized = False
        self._username_cache: dict[str, str] = {}  # address -> username

    def add_callback(self, callback: callable):
        """Add callback for new trade notifications"""
        self._callbacks.append(callback)

    async def _ensure_initialized(self):
        """Load wallets from database on first access"""
        if self._initialized:
            return

        async with AsyncSessionLocal() as session:
            result = await session.execute(select(TrackedWallet))
            db_wallets = result.scalars().all()

            for wallet in db_wallets:
                self.tracked_wallets[wallet.address.lower()] = {
                    "address": wallet.address,
                    "label": wallet.label or wallet.address[:10] + "...",
                    "username": None,
                    "last_trade_id": None,
                    "positions": [],
                    "recent_trades": [],
                    "added_at": wallet.added_at.isoformat()
                    if wallet.added_at
                    else None,
                }

        self._initialized = True

    async def _lookup_username(self, address: str) -> Optional[str]:
        """Look up Polymarket username for an address, with caching."""
        address_lower = address.lower()
        if address_lower in self._username_cache:
            return self._username_cache[address_lower]

        try:
            profile = await self.client.get_user_profile(address)
            username = profile.get("username")
            if username:
                self._username_cache[address_lower] = username
                return username
        except Exception as e:
            print(f"Username lookup failed for {address}: {e}")

        return None

    async def add_wallet(
        self,
        address: str,
        label: str = None,
        *,
        fetch_initial: bool = True,
    ):
        """Add a wallet to track (persisted to database).

        Parameters
        ----------
        address : str
            Wallet address to track.
        label : str | None
            Optional display label.
        fetch_initial : bool
            When True, fetch username/positions/trades immediately.
            Group operations can set this to False to avoid expensive
            N-wallet bootstrap calls in a single request.
        """
        await self._ensure_initialized()
        address_lower = address.lower()

        # Add to database
        async with AsyncSessionLocal() as session:
            existing = await session.get(TrackedWallet, address_lower)
            if not existing:
                wallet = TrackedWallet(
                    address=address_lower, label=label or address[:10] + "..."
                )
                session.add(wallet)
                await session.commit()

        username = None
        if fetch_initial:
            username = await self._lookup_username(address)

        # Add to in-memory cache
        self.tracked_wallets[address_lower] = {
            "address": address,
            "label": label or address[:10] + "...",
            "username": username,
            "last_trade_id": None,
            "positions": [],
            "recent_trades": [],
        }

        # Fetch initial state only when requested.
        if fetch_initial:
            await self._update_wallet(address)

    async def remove_wallet(self, address: str):
        """Remove a wallet from tracking"""
        address_lower = address.lower()

        # Remove from database
        async with AsyncSessionLocal() as session:
            wallet = await session.get(TrackedWallet, address_lower)
            if wallet:
                await session.delete(wallet)
                await session.commit()

        # Remove from in-memory cache
        self.tracked_wallets.pop(address_lower, None)

    async def _update_wallet(self, address: str) -> list[dict]:
        """Update wallet state and return new trades"""
        address_lower = address.lower()
        wallet = self.tracked_wallets.get(address_lower)
        if not wallet:
            return []

        new_trades = []

        try:
            # Refresh username if not cached yet
            if not wallet.get("username"):
                username = await self._lookup_username(address)
                if username:
                    wallet["username"] = username

            # Fetch positions
            positions = await self.client.get_wallet_positions(address)
            wallet["positions"] = positions

            # Fetch recent trades
            trades = await self.client.get_wallet_trades(address, limit=200)
            wallet["recent_trades"] = trades

            # Check for new trades
            last_id = wallet.get("last_trade_id")
            if last_id and trades:
                new_trades = [t for t in trades if t.get("id", "") > last_id]

            # Update last trade ID
            if trades:
                wallet["last_trade_id"] = trades[0].get("id", "")

        except Exception as e:
            print(f"Error updating wallet {address}: {e}")

        return new_trades

    async def check_all_wallets(self) -> list[dict]:
        """Check all tracked wallets for new activity (concurrent)."""
        all_new_trades = []

        # Update wallets concurrently instead of sequentially to avoid
        # blocking the event loop for extended periods when many wallets
        # are tracked.
        addresses = list(self.tracked_wallets.keys())
        if addresses:
            results = await asyncio.gather(
                *[self._update_wallet(addr) for addr in addresses],
                return_exceptions=True,
            )
            for address, result in zip(addresses, results):
                if isinstance(result, Exception):
                    print(f"  Wallet update error for {address}: {result}")
                    continue
                if result:
                    wallet = self.tracked_wallets.get(address)
                    if wallet:
                        for trade in result:
                            trade["_wallet_label"] = wallet["label"]
                            trade["_wallet_address"] = address
                        all_new_trades.extend(result)

        # Notify callbacks
        for trade in all_new_trades:
            for callback in self._callbacks:
                try:
                    await callback(trade)
                except Exception as e:
                    print(f"Wallet callback error: {e}")

        return all_new_trades

    async def start_monitoring(self, interval_seconds: int = 30):
        """Start continuous wallet monitoring"""
        self._running = True
        print(f"Starting wallet monitor (interval: {interval_seconds}s)")

        while self._running:
            if not global_pause_state.is_paused:
                try:
                    new_trades = await self.check_all_wallets()
                    if new_trades:
                        print(
                            f"[{utcnow().isoformat()}] {len(new_trades)} new trades detected"
                        )
                except Exception as e:
                    print(f"Wallet monitor error: {e}")

            await asyncio.sleep(interval_seconds)

    def stop(self):
        """Stop wallet monitoring"""
        self._running = False

    async def get_wallet_info(self, address: str) -> Optional[dict]:
        """Get current info for a wallet"""
        await self._ensure_initialized()
        return self.tracked_wallets.get(address.lower())

    async def get_all_wallets(self) -> list[dict]:
        """Get all tracked wallets (returns cached data).

        Data is refreshed in the background by the monitoring loop. This
        method returns immediately so API endpoints stay responsive.
        """
        await self._ensure_initialized()
        return list(self.tracked_wallets.values())


# Singleton instance
wallet_tracker = WalletTracker()
