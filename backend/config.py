from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    # API Base URLs
    GAMMA_API_URL: str = "https://gamma-api.polymarket.com"
    CLOB_API_URL: str = "https://clob.polymarket.com"
    DATA_API_URL: str = "https://data-api.polymarket.com"

    # WebSocket URLs
    CLOB_WS_URL: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

    # Scanner Settings
    SCAN_INTERVAL_SECONDS: int = 60
    MIN_PROFIT_THRESHOLD: float = 0.025  # 2.5% minimum profit after fees
    POLYMARKET_FEE: float = 0.02  # 2% winner fee

    # Market Settings
    MAX_MARKETS_TO_SCAN: int = 500
    MIN_LIQUIDITY: float = 1000.0  # Minimum liquidity in USD

    # Wallet Tracking
    TRACKED_WALLETS: list[str] = []

    # Notifications
    TELEGRAM_BOT_TOKEN: Optional[str] = None
    TELEGRAM_CHAT_ID: Optional[str] = None

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./arbitrage.db"

    class Config:
        env_file = ".env"


settings = Settings()
