"""
Pre-Flight Configuration Validator

Validates all configuration parameters before the bot starts trading.
Catches missing credentials, invalid formats, and out-of-range parameters
at startup instead of discovering them at runtime.

Inspired by terauss/Polymarket-Copy-Trading-Bot validate_setup binary.
"""

import re
from dataclasses import dataclass, field
from typing import Optional
from utils.logger import get_logger

logger = get_logger("config_validator")


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checks_passed: int = 0
    checks_failed: int = 0
    checks_warned: int = 0


class ConfigValidator:
    """Validates configuration parameters before trading begins."""

    def validate_all(self, settings) -> ValidationResult:
        """Run all validation checks against settings object."""
        result = ValidationResult(valid=True)

        # Core API URLs
        self._check_url(result, "GAMMA_API_URL", settings.GAMMA_API_URL)
        self._check_url(result, "CLOB_API_URL", settings.CLOB_API_URL)

        # Scanner parameters
        self._check_range(result, "SCAN_INTERVAL_SECONDS", settings.SCAN_INTERVAL_SECONDS, 10, 3600)
        self._check_range(result, "MIN_PROFIT_THRESHOLD", settings.MIN_PROFIT_THRESHOLD, 0.001, 0.5)
        self._check_range(result, "MAX_MARKETS_TO_SCAN", settings.MAX_MARKETS_TO_SCAN, 1, 10000)
        self._check_range(result, "MAX_EVENTS_TO_SCAN", settings.MAX_EVENTS_TO_SCAN, 1, 10000)
        self._check_range(result, "MARKET_FETCH_PAGE_SIZE", settings.MARKET_FETCH_PAGE_SIZE, 50, 500)
        self._check_range(result, "MIN_LIQUIDITY", settings.MIN_LIQUIDITY, 0, 1_000_000)

        # Trading safety limits
        self._check_range(result, "MAX_TRADE_SIZE_USD", settings.MAX_TRADE_SIZE_USD, 0.1, 100_000)
        self._check_range(
            result,
            "MAX_DAILY_TRADE_VOLUME",
            settings.MAX_DAILY_TRADE_VOLUME,
            1,
            10_000_000,
        )
        self._check_range(result, "MAX_SLIPPAGE_PERCENT", settings.MAX_SLIPPAGE_PERCENT, 0.1, 50)
        self._check_range(result, "MIN_ORDER_SIZE_USD", settings.MIN_ORDER_SIZE_USD, 0.01, 1000)

        if settings.MIN_ORDER_SIZE_USD >= settings.MAX_TRADE_SIZE_USD:
            self._add_error(result, "MIN_ORDER_SIZE_USD must be less than MAX_TRADE_SIZE_USD")

        # Circuit breaker
        self._check_range(
            result,
            "CB_LARGE_TRADE_SHARES",
            settings.CB_LARGE_TRADE_SHARES,
            1,
            1_000_000,
        )
        self._check_range(result, "CB_CONSECUTIVE_TRIGGER", settings.CB_CONSECUTIVE_TRIGGER, 1, 100)
        self._check_range(
            result,
            "CB_DETECTION_WINDOW_SECONDS",
            settings.CB_DETECTION_WINDOW_SECONDS,
            1,
            600,
        )
        self._check_range(
            result,
            "CB_TRIP_DURATION_SECONDS",
            settings.CB_TRIP_DURATION_SECONDS,
            1,
            3600,
        )

        # Depth analysis
        self._check_range(result, "MIN_DEPTH_USD", settings.MIN_DEPTH_USD, 1, 100_000)

        # Trading credentials (optional - warns if partially configured)
        has_private_key = bool(settings.POLYMARKET_PRIVATE_KEY)
        has_api_creds = bool(
            settings.POLYMARKET_API_KEY and settings.POLYMARKET_API_SECRET and settings.POLYMARKET_API_PASSPHRASE
        )
        if has_private_key and not has_api_creds:
            self._add_warning(result, "POLYMARKET_PRIVATE_KEY set but API credentials (KEY/SECRET/PASSPHRASE) missing")
        elif has_api_creds and not has_private_key:
            self._add_warning(result, "Polymarket API credentials set but POLYMARKET_PRIVATE_KEY missing")

        # Telegram (optional but warn if partially configured)
        has_token = bool(settings.TELEGRAM_BOT_TOKEN)
        has_chat = bool(settings.TELEGRAM_CHAT_ID)
        if has_token != has_chat:
            self._add_warning(result, "Telegram partially configured: need both BOT_TOKEN and CHAT_ID")

        # Wallet tracking
        for wallet in settings.TRACKED_WALLETS:
            if wallet and not self._is_valid_eth_address(wallet):
                self._add_error(result, f"Invalid wallet address: {wallet}")

        # Database
        if not settings.DATABASE_URL:
            self._add_error(result, "DATABASE_URL is required")

        result.valid = len(result.errors) == 0

        log_method = logger.info if result.valid else logger.error
        log_method(
            "Config validation complete",
            valid=result.valid,
            passed=result.checks_passed,
            failed=result.checks_failed,
            warnings=result.checks_warned,
        )

        return result

    def validate_trading_ready(self, settings) -> ValidationResult:
        """Stricter validation specifically for enabling live trading."""
        result = self.validate_all(settings)

        if not settings.POLYMARKET_PRIVATE_KEY:
            self._add_error(result, "POLYMARKET_PRIVATE_KEY required for live trading")
        elif not settings.POLYMARKET_API_KEY:
            self._add_warning(result, "API keys not set - will attempt auto-generation via EIP-712")

        result.valid = len(result.errors) == 0
        return result

    # --- Validation helpers ---

    def _check_url(self, result: ValidationResult, name: str, value: str):
        if not value:
            self._add_error(result, f"{name} is required")
        elif not value.startswith(("http://", "https://", "wss://")):
            self._add_error(result, f"{name} must be a valid URL (got: {value})")
        else:
            result.checks_passed += 1

    def _check_range(self, result: ValidationResult, name: str, value, min_val, max_val):
        if value < min_val or value > max_val:
            self._add_error(result, f"{name}={value} out of range [{min_val}, {max_val}]")
        else:
            result.checks_passed += 1

    def _check_required(self, result: ValidationResult, name: str, value: Optional[str]):
        if not value:
            self._add_error(result, f"{name} is required")
        else:
            result.checks_passed += 1

    def _check_private_key(self, result: ValidationResult, key: Optional[str]):
        if not key:
            self._add_error(result, "POLYMARKET_PRIVATE_KEY is required for trading")
            return
        cleaned = key.replace("0x", "")
        if len(cleaned) != 64 or not all(c in "0123456789abcdefABCDEF" for c in cleaned):
            self._add_error(
                result,
                "POLYMARKET_PRIVATE_KEY must be 64 hex characters (with or without 0x prefix)",
            )
        else:
            result.checks_passed += 1

    def _is_valid_eth_address(self, address: str) -> bool:
        return bool(re.match(r"^0x[0-9a-fA-F]{40}$", address))

    def _add_error(self, result: ValidationResult, msg: str):
        result.errors.append(msg)
        result.checks_failed += 1
        logger.error(f"Validation FAILED: {msg}")

    def _add_warning(self, result: ValidationResult, msg: str):
        result.warnings.append(msg)
        result.checks_warned += 1
        logger.warning(f"Validation WARNING: {msg}")


config_validator = ConfigValidator()
