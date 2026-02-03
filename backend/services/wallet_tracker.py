import asyncio
from datetime import datetime
from typing import Optional

from config import settings
from services.polymarket import polymarket_client


class WalletTracker:
    """Track specific wallets for trade activity"""

    def __init__(self):
        self.client = polymarket_client
        self.tracked_wallets: dict[str, dict] = {}
        self._running = False
        self._callbacks: list[callable] = []

    def add_callback(self, callback: callable):
        """Add callback for new trade notifications"""
        self._callbacks.append(callback)

    async def add_wallet(self, address: str, label: str = None):
        """Add a wallet to track"""
        self.tracked_wallets[address.lower()] = {
            "address": address,
            "label": label or address[:10] + "...",
            "last_trade_id": None,
            "positions": [],
            "recent_trades": []
        }

        # Fetch initial state
        await self._update_wallet(address)

    def remove_wallet(self, address: str):
        """Remove a wallet from tracking"""
        self.tracked_wallets.pop(address.lower(), None)

    async def _update_wallet(self, address: str) -> list[dict]:
        """Update wallet state and return new trades"""
        address_lower = address.lower()
        wallet = self.tracked_wallets.get(address_lower)
        if not wallet:
            return []

        new_trades = []

        try:
            # Fetch positions
            positions = await self.client.get_wallet_positions(address)
            wallet["positions"] = positions

            # Fetch recent trades
            trades = await self.client.get_wallet_trades(address, limit=50)
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
        """Check all tracked wallets for new activity"""
        all_new_trades = []

        for address in list(self.tracked_wallets.keys()):
            new_trades = await self._update_wallet(address)
            if new_trades:
                wallet = self.tracked_wallets[address]
                for trade in new_trades:
                    trade["_wallet_label"] = wallet["label"]
                    trade["_wallet_address"] = address
                all_new_trades.extend(new_trades)

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
            try:
                new_trades = await self.check_all_wallets()
                if new_trades:
                    print(f"[{datetime.utcnow().isoformat()}] {len(new_trades)} new trades detected")
            except Exception as e:
                print(f"Wallet monitor error: {e}")

            await asyncio.sleep(interval_seconds)

    def stop(self):
        """Stop wallet monitoring"""
        self._running = False

    def get_wallet_info(self, address: str) -> Optional[dict]:
        """Get current info for a wallet"""
        return self.tracked_wallets.get(address.lower())

    def get_all_wallets(self) -> list[dict]:
        """Get all tracked wallets"""
        return list(self.tracked_wallets.values())


# Singleton instance
wallet_tracker = WalletTracker()
