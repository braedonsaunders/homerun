"""
Strategy SDK — high-level helpers for custom strategy development.

This module provides a clean, stable API surface for strategy authors.
Import it in your strategy code with:

    from services.strategy_sdk import StrategySDK

All methods are designed to be safe and handle missing data gracefully.
"""

from __future__ import annotations

import json
import logging
import re
import warnings
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class StrategySDK:
    """High-level helpers for strategy authors.

    All methods are static or class methods — no instantiation needed.

    Usage in opportunity strategies (detect method):
        from services.strategy_sdk import StrategySDK

        price = StrategySDK.get_live_price(market, prices)
        spread = StrategySDK.get_spread_bps(market, prices)

    Usage for LLM calls (async):
        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            StrategySDK.ask_llm("Analyze this market situation...")
        )
    """

    _CONFIG_SLUG_RE = re.compile(r"^[a-z][a-z0-9_]{1,48}[a-z0-9]$")
    TRADER_TIER_CANONICAL = ("low", "medium", "high", "extreme")
    TRADER_SIDE_CANONICAL = ("all", "buy", "sell")
    TRADER_SOURCE_SCOPE_CANONICAL = ("all", "tracked", "pool")
    TRADER_FILTER_DEFAULTS: dict[str, Any] = {
        "min_confidence": 0.45,
        "min_tier": "high",
        "min_wallet_count": 2,
        "max_entry_price": 0.85,
        "firehose_require_tradable_market": True,
        "firehose_exclude_crypto_markets": True,
        "firehose_require_qualified_source": True,
        "firehose_max_age_minutes": 180,
        "firehose_source_scope": "all",
        "firehose_side_filter": "all",
    }
    TRADER_FILTER_CONFIG_SCHEMA: dict[str, Any] = {
        "param_fields": [
            {"key": "min_confidence", "label": "Min Confidence", "type": "number", "min": 0, "max": 1},
            {
                "key": "min_tier",
                "label": "Min Tier",
                "type": "enum",
                "options": ["low", "medium", "high", "extreme"],
            },
            {"key": "min_wallet_count", "label": "Min Wallet Count", "type": "integer", "min": 1},
            {"key": "max_entry_price", "label": "Max Entry Price", "type": "number", "min": 0, "max": 1},
            {"key": "firehose_require_tradable_market", "label": "Require Tradable Market", "type": "boolean"},
            {"key": "firehose_exclude_crypto_markets", "label": "Exclude Crypto Markets", "type": "boolean"},
            {"key": "firehose_require_qualified_source", "label": "Require Qualified Source", "type": "boolean"},
            {
                "key": "firehose_max_age_minutes",
                "label": "Firehose Max Age (min)",
                "type": "integer",
                "min": 1,
                "max": 1440,
            },
            {
                "key": "firehose_source_scope",
                "label": "Source Scope",
                "type": "enum",
                "options": ["all", "tracked", "pool"],
            },
            {
                "key": "firehose_side_filter",
                "label": "Side Filter",
                "type": "enum",
                "options": ["all", "buy", "sell"],
            },
        ]
    }

    @staticmethod
    def _normalize_strategy_slug(slug: str) -> str:
        value = str(slug or "").strip().lower()
        if not value:
            return ""
        if StrategySDK._CONFIG_SLUG_RE.match(value):
            return value
        value = re.sub(r"[^a-z0-9_]", "_", value)
        value = value.strip("_")
        if not value:
            return ""
        if not value[0].isalpha():
            value = f"s_{value}"
        if len(value) > 50:
            value = value[:50].rstrip("_")
        if value and not value[-1].isalnum():
            value = f"{value}0"
        if StrategySDK._CONFIG_SLUG_RE.match(value):
            return value
        return ""

    @staticmethod
    def _coerce_bool(value: Any, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        normalized = str(value).strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        return default

    @staticmethod
    def _coerce_float(value: Any, default: float, lo: float, hi: float) -> float:
        try:
            parsed = float(value)
        except Exception:
            parsed = default
        if parsed != parsed or parsed in (float("inf"), float("-inf")):
            parsed = default
        return max(lo, min(hi, parsed))

    @staticmethod
    def _coerce_int(value: Any, default: int, lo: int, hi: int) -> int:
        try:
            parsed = int(float(value))
        except Exception:
            parsed = default
        return max(lo, min(hi, parsed))

    @staticmethod
    def normalize_trader_tier(value: Any, default: str = "low") -> str:
        normalized = str(value or "").strip().lower()
        aliases = {
            "watch": "low",
            "low": "low",
            "medium": "medium",
            "mid": "medium",
            "high": "high",
            "extreme": "extreme",
        }
        tier = aliases.get(normalized, "")
        if tier in StrategySDK.TRADER_TIER_CANONICAL:
            return tier
        fallback = str(default or "low").strip().lower()
        if fallback in StrategySDK.TRADER_TIER_CANONICAL:
            return fallback
        return "low"

    @staticmethod
    def normalize_trader_side(value: Any, default: str = "all") -> str:
        normalized = str(value or "").strip().lower()
        aliases = {
            "all": "all",
            "buy": "buy",
            "buy_yes": "buy",
            "yes": "buy",
            "long": "buy",
            "sell": "sell",
            "buy_no": "sell",
            "no": "sell",
            "short": "sell",
        }
        side = aliases.get(normalized, "")
        if side in StrategySDK.TRADER_SIDE_CANONICAL:
            return side
        fallback = str(default or "all").strip().lower()
        if fallback in StrategySDK.TRADER_SIDE_CANONICAL:
            return fallback
        return "all"

    @staticmethod
    def normalize_trader_source_scope(value: Any, default: str = "all") -> str:
        normalized = str(value or "").strip().lower()
        aliases = {
            "all": "all",
            "confluence": "all",
            "tracked": "tracked",
            "pool": "pool",
            "insider": "pool",
        }
        scope = aliases.get(normalized, "")
        if scope in StrategySDK.TRADER_SOURCE_SCOPE_CANONICAL:
            return scope
        fallback = str(default or "all").strip().lower()
        if fallback in StrategySDK.TRADER_SOURCE_SCOPE_CANONICAL:
            return fallback
        return "all"

    @staticmethod
    def normalize_trader_source_flags(value: Any) -> dict[str, bool]:
        flags = value if isinstance(value, dict) else {}
        from_pool = bool(flags.get("from_pool"))
        from_tracked = bool(flags.get("from_tracked_traders"))
        from_groups = bool(flags.get("from_trader_groups"))
        qualified = bool(flags.get("qualified", from_pool or from_tracked or from_groups))
        return {
            "from_pool": from_pool,
            "from_tracked_traders": from_tracked,
            "from_trader_groups": from_groups,
            "qualified": qualified,
        }

    @staticmethod
    def infer_trader_side(signal: dict[str, Any]) -> str:
        raw_direction = StrategySDK.normalize_trader_side(signal.get("direction"), default="")
        if raw_direction:
            return raw_direction
        raw_outcome = StrategySDK.normalize_trader_side(signal.get("outcome"), default="")
        if raw_outcome:
            return raw_outcome
        raw_signal_type = StrategySDK.normalize_trader_side(signal.get("signal_type"), default="")
        if raw_signal_type:
            return raw_signal_type
        return "all"

    @staticmethod
    def normalize_trader_signal(signal: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(signal or {})
        raw_tier = normalized.get("tier")
        normalized["tier_raw"] = raw_tier
        normalized["tier"] = StrategySDK.normalize_trader_tier(raw_tier)
        normalized["side"] = StrategySDK.infer_trader_side(normalized)
        normalized["source_flags"] = StrategySDK.normalize_trader_source_flags(normalized.get("source_flags"))
        return normalized

    @staticmethod
    def trader_filter_defaults() -> dict[str, Any]:
        return dict(StrategySDK.TRADER_FILTER_DEFAULTS)

    @staticmethod
    def trader_filter_config_schema() -> dict[str, Any]:
        return dict(StrategySDK.TRADER_FILTER_CONFIG_SCHEMA)

    @staticmethod
    def validate_trader_filter_config(config: Any) -> dict[str, Any]:
        cfg = dict(StrategySDK.TRADER_FILTER_DEFAULTS)
        if isinstance(config, dict):
            cfg.update({str(k): v for k, v in config.items() if str(k) != "_schema"})

        cfg["min_confidence"] = StrategySDK._coerce_float(cfg.get("min_confidence"), 0.45, 0.0, 1.0)
        cfg["min_tier"] = StrategySDK.normalize_trader_tier(cfg.get("min_tier"), default="high")
        cfg["min_wallet_count"] = StrategySDK._coerce_int(cfg.get("min_wallet_count"), 2, 1, 1000)
        cfg["max_entry_price"] = StrategySDK._coerce_float(cfg.get("max_entry_price"), 0.85, 0.0, 1.0)
        cfg["firehose_require_tradable_market"] = StrategySDK._coerce_bool(
            cfg.get("firehose_require_tradable_market"),
            True,
        )
        cfg["firehose_exclude_crypto_markets"] = StrategySDK._coerce_bool(
            cfg.get("firehose_exclude_crypto_markets"),
            True,
        )
        cfg["firehose_require_qualified_source"] = StrategySDK._coerce_bool(
            cfg.get("firehose_require_qualified_source"),
            True,
        )
        cfg["firehose_max_age_minutes"] = StrategySDK._coerce_int(
            cfg.get("firehose_max_age_minutes"),
            180,
            0,
            1440,
        )
        cfg["firehose_source_scope"] = StrategySDK.normalize_trader_source_scope(
            cfg.get("firehose_source_scope"),
            default="all",
        )
        cfg["firehose_side_filter"] = StrategySDK.normalize_trader_side(
            cfg.get("firehose_side_filter"),
            default="all",
        )
        return cfg

    @staticmethod
    def _strategy_config_dir() -> Path:
        return Path(__file__).resolve().parents[1] / "strategy_configs"

    @staticmethod
    def _strategy_config_path(slug: str) -> Optional[Path]:
        normalized = StrategySDK._normalize_strategy_slug(slug)
        if not normalized:
            return None
        return StrategySDK._strategy_config_dir() / f"{normalized}.json"

    @staticmethod
    def get_strategy_config_path(slug: str) -> Optional[str]:
        """Return the absolute JSON config path for a strategy slug."""
        path = StrategySDK._strategy_config_path(slug)
        if path is None:
            return None
        return str(path)

    @staticmethod
    def get_strategy_config_mtime_ns(slug: str) -> int:
        """Return config file mtime (ns), or 0 when no file exists."""
        path = StrategySDK._strategy_config_path(slug)
        if path is None or not path.exists():
            return 0
        try:
            return int(path.stat().st_mtime_ns)
        except Exception:
            return 0

    @staticmethod
    def read_strategy_config_file(slug: str) -> dict[str, Any]:
        """Read JSON config overrides from backend/strategy_configs/<slug>.json.

        .. deprecated::
            File-based config overrides are no longer applied by the strategy
            loader. Config is now managed exclusively via the DB (UI). This
            function is retained for compatibility but has no effect on strategy
            runtime config.
        """
        warnings.warn(
            "read_strategy_config_file() is deprecated. Config is now managed exclusively via the DB. "
            "File-based overrides are no longer applied by the strategy loader.",
            DeprecationWarning,
            stacklevel=2,
        )
        path = StrategySDK._strategy_config_path(slug)
        if path is None or not path.exists():
            return {}
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(parsed, dict):
                return parsed
            logger.warning("Strategy config file is not an object: %s", path)
            return {}
        except Exception as e:
            logger.warning("Failed to read strategy config file %s: %s", path, e)
            return {}

    @staticmethod
    def write_strategy_config_file(slug: str, config: dict[str, Any]) -> bool:
        """Write config overrides to backend/strategy_configs/<slug>.json."""
        path = StrategySDK._strategy_config_path(slug)
        if path is None:
            return False
        payload = config if isinstance(config, dict) else {}
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            encoded = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True)
            path.write_text(f"{encoded}\n", encoding="utf-8")
            return True
        except Exception as e:
            logger.warning("Failed to write strategy config file %s: %s", path, e)
            return False

    @staticmethod
    def rename_strategy_config_file(old_slug: str, new_slug: str) -> bool:
        """Rename a strategy config file when strategy slug changes."""
        old_path = StrategySDK._strategy_config_path(old_slug)
        new_path = StrategySDK._strategy_config_path(new_slug)
        if old_path is None or new_path is None or old_path == new_path:
            return True
        if not old_path.exists():
            return True
        try:
            new_path.parent.mkdir(parents=True, exist_ok=True)
            old_path.replace(new_path)
            return True
        except Exception as e:
            logger.warning(
                "Failed to rename strategy config file %s -> %s: %s",
                old_path,
                new_path,
                e,
            )
            return False

    @staticmethod
    def delete_strategy_config_file(slug: str) -> bool:
        """Delete backend/strategy_configs/<slug>.json if it exists."""
        path = StrategySDK._strategy_config_path(slug)
        if path is None or not path.exists():
            return True
        try:
            path.unlink()
            return True
        except Exception as e:
            logger.warning("Failed to delete strategy config file %s: %s", path, e)
            return False

    # ── Price helpers ──────────────────────────────────────────────

    @staticmethod
    def get_live_price(market: Any, prices: dict[str, dict], side: str = "YES") -> float:
        """Get the best available mid price for a market.

        Uses live CLOB prices when available, falls back to API prices.

        Args:
            market: A Market object with clob_token_ids and outcome_prices.
            prices: The prices dict passed to detect().
            side: 'YES' or 'NO'.

        Returns:
            The mid price as a float (0.0 if unavailable).
        """
        idx = 0 if side.upper() == "YES" else 1
        fallback = 0.0
        if hasattr(market, "outcome_prices") and len(market.outcome_prices) > idx:
            fallback = market.outcome_prices[idx]

        token_ids = getattr(market, "clob_token_ids", None) or []
        if len(token_ids) > idx:
            token_id = token_ids[idx]
            if token_id in prices:
                return prices[token_id].get("mid", fallback)
        return fallback

    @staticmethod
    def get_spread_bps(market: Any, prices: dict[str, dict], side: str = "YES") -> Optional[float]:
        """Get the bid-ask spread in basis points for a market side.

        Args:
            market: A Market object.
            prices: The prices dict passed to detect().
            side: 'YES' or 'NO'.

        Returns:
            Spread in basis points, or None if data unavailable.
        """
        idx = 0 if side.upper() == "YES" else 1
        token_ids = getattr(market, "clob_token_ids", None) or []
        if len(token_ids) <= idx:
            return None
        token_id = token_ids[idx]
        data = prices.get(token_id)
        if not data:
            return None
        bid = data.get("best_bid", 0)
        ask = data.get("best_ask", 0)
        mid = data.get("mid", 0)
        if mid <= 0 or bid <= 0 or ask <= 0:
            return None
        return ((ask - bid) / mid) * 10000

    @staticmethod
    def get_ws_mid_price(token_id: str) -> Optional[float]:
        """Get the real-time WebSocket mid price for a token.

        Args:
            token_id: The CLOB token ID.

        Returns:
            Mid price or None if not fresh.
        """
        try:
            from services.ws_feeds import get_feed_manager

            fm = get_feed_manager()
            if fm and fm.cache.is_fresh(token_id):
                return fm.cache.get_mid_price(token_id)
        except Exception:
            pass
        return None

    @staticmethod
    def get_ws_spread_bps(token_id: str) -> Optional[float]:
        """Get the real-time WebSocket spread in basis points.

        Args:
            token_id: The CLOB token ID.

        Returns:
            Spread in bps or None.
        """
        try:
            from services.ws_feeds import get_feed_manager

            fm = get_feed_manager()
            if fm:
                return fm.cache.get_spread_bps(token_id)
        except Exception:
            pass
        return None

    # ── Chainlink oracle ──────────────────────────────────────────

    @staticmethod
    def get_chainlink_price(asset: str) -> Optional[float]:
        """Get the latest Chainlink oracle price for a crypto asset.

        Args:
            asset: Asset symbol — 'BTC', 'ETH', 'SOL', or 'XRP'.

        Returns:
            The oracle price in USD, or None if unavailable.
        """
        try:
            from services.chainlink_feed import get_chainlink_feed

            feed = get_chainlink_feed()
            if feed:
                p = feed.get_price(asset.upper())
                return p.price if p else None
        except Exception:
            pass
        return None

    # ── LLM helpers ───────────────────────────────────────────────

    @staticmethod
    async def ask_llm(
        prompt: str,
        model: str = "gpt-4o-mini",
        system: str = "",
        purpose: str = "custom_strategy",
    ) -> str:
        """Send a prompt to an LLM and get the text response.

        Args:
            prompt: The user prompt.
            model: Model identifier (e.g. 'gpt-4o-mini', 'claude-haiku-4-5-20251001').
            system: Optional system prompt.
            purpose: Usage attribution tag (default: 'custom_strategy').

        Returns:
            The LLM response text, or empty string on failure.
        """
        try:
            from services.ai import get_llm_manager
            from services.ai.llm_provider import LLMMessage

            manager = get_llm_manager()
            if not manager.is_available():
                return ""

            messages = []
            if system:
                messages.append(LLMMessage(role="system", content=system))
            messages.append(LLMMessage(role="user", content=prompt))

            response = await manager.chat(
                messages=messages,
                model=model,
                purpose=purpose,
            )
            return response.content
        except Exception as e:
            logger.warning("StrategySDK.ask_llm failed: %s", e)
            return ""

    @staticmethod
    async def ask_llm_json(
        prompt: str,
        schema: dict,
        model: str = "gpt-4o-mini",
        system: str = "",
        purpose: str = "custom_strategy",
    ) -> dict:
        """Get structured JSON output from an LLM.

        Args:
            prompt: The user prompt.
            schema: JSON Schema the response must conform to.
            model: Model identifier.
            system: Optional system prompt.
            purpose: Usage attribution tag.

        Returns:
            Parsed dict conforming to the schema, or empty dict on failure.
        """
        try:
            from services.ai import get_llm_manager
            from services.ai.llm_provider import LLMMessage

            manager = get_llm_manager()
            if not manager.is_available():
                return {}

            messages = []
            if system:
                messages.append(LLMMessage(role="system", content=system))
            messages.append(LLMMessage(role="user", content=prompt))

            return await manager.structured_output(
                messages=messages,
                schema=schema,
                model=model,
                purpose=purpose,
            )
        except Exception as e:
            logger.warning("StrategySDK.ask_llm_json failed: %s", e)
            return {}

    # ── Fee calculation ───────────────────────────────────────────

    @staticmethod
    def calculate_fees(
        total_cost: float,
        expected_payout: float = 1.0,
        n_legs: int = 1,
    ) -> Optional[dict]:
        """Calculate comprehensive fees for a potential trade.

        Args:
            total_cost: Total cost to enter the position.
            expected_payout: Expected payout if the trade succeeds.
            n_legs: Number of legs in the trade.

        Returns:
            Dict with fee breakdown, or None on failure.
        """
        try:
            from services.fee_model import fee_model

            breakdown = fee_model.full_breakdown(
                total_cost=total_cost,
                expected_payout=expected_payout,
                n_legs=n_legs,
            )
            return {
                "winner_fee": breakdown.winner_fee,
                "gas_cost_usd": breakdown.gas_cost_usd,
                "spread_cost": breakdown.spread_cost,
                "multi_leg_slippage": breakdown.multi_leg_slippage,
                "total_fees": breakdown.total_fees,
                "fee_as_pct_of_payout": breakdown.fee_as_pct_of_payout,
            }
        except Exception as e:
            logger.warning("StrategySDK.calculate_fees failed: %s", e)
            return None

    # ── Market filtering helpers ──────────────────────────────────

    @staticmethod
    def active_binary_markets(markets: list) -> list:
        """Filter to only active, non-closed, binary (2-outcome) markets.

        Args:
            markets: List of Market objects.

        Returns:
            Filtered list of markets.
        """
        return [
            m
            for m in markets
            if getattr(m, "active", False)
            and not getattr(m, "closed", False)
            and len(getattr(m, "outcome_prices", []) or []) == 2
        ]

    @staticmethod
    def markets_by_category(events: list, category: str) -> list:
        """Get all active markets belonging to events of a given category.

        Args:
            events: List of Event objects.
            category: Category to filter by (e.g. 'crypto', 'politics').

        Returns:
            List of Market objects in matching events.
        """
        result = []
        for event in events:
            if getattr(event, "category", None) == category:
                for market in getattr(event, "markets", []) or []:
                    if getattr(market, "active", False) and not getattr(market, "closed", False):
                        result.append(market)
        return result

    @staticmethod
    def find_event_for_market(market: Any, events: list) -> Optional[Any]:
        """Find the parent Event for a given Market.

        Args:
            market: A Market object.
            events: List of Event objects.

        Returns:
            The parent Event, or None.
        """
        market_id = getattr(market, "id", None)
        if not market_id:
            return None
        for event in events:
            for m in getattr(event, "markets", []) or []:
                if getattr(m, "id", None) == market_id:
                    return event
        return None

    # ── Order book depth ───────────────────────────────────────

    @staticmethod
    def get_order_book_depth(
        market: Any,
        side: str = "YES",
        size_usd: float = 100.0,
    ) -> Optional[dict]:
        """Get order book depth analysis for a market side.

        Uses the internal VWAP calculator to estimate execution cost,
        slippage, and fill probability for a given position size.

        Args:
            market: A Market object.
            side: 'YES' or 'NO'.
            size_usd: Position size in USD to estimate execution for.

        Returns:
            Dict with depth analysis, or None if unavailable:
            {
                "vwap_price": float,       # Volume-weighted avg price
                "slippage_bps": float,     # Slippage in basis points
                "fill_probability": float, # Estimated fill probability (0-1)
                "available_liquidity": float,  # Liquidity at this level
            }
        """
        try:
            from services.optimization.vwap import vwap_calculator

            if not vwap_calculator:
                return None

            token_idx = 0 if side.upper() == "YES" else 1
            token_ids = getattr(market, "clob_token_ids", None) or []
            if len(token_ids) <= token_idx:
                return None

            result = vwap_calculator.estimate_execution(
                token_id=token_ids[token_idx],
                size_usd=size_usd,
            )
            if result is None:
                return None

            return {
                "vwap_price": result.get("vwap_price", 0),
                "slippage_bps": result.get("slippage_bps", 0),
                "fill_probability": result.get("fill_probability", 1.0),
                "available_liquidity": result.get("available_liquidity", 0),
            }
        except Exception:
            return None

    @staticmethod
    def get_book_levels(
        market: Any,
        side: str = "YES",
        max_levels: int = 10,
    ) -> Optional[list[dict]]:
        """Get raw order book levels (bids or asks) for a market side.

        Args:
            market: A Market object.
            side: 'YES' or 'NO'.
            max_levels: Maximum number of price levels to return.

        Returns:
            List of {price, size} dicts sorted by best price, or None.
        """
        try:
            from services.optimization.vwap import vwap_calculator

            if not vwap_calculator:
                return None

            token_idx = 0 if side.upper() == "YES" else 1
            token_ids = getattr(market, "clob_token_ids", None) or []
            if len(token_ids) <= token_idx:
                return None

            return vwap_calculator.get_book_levels(
                token_id=token_ids[token_idx],
                max_levels=max_levels,
            )
        except Exception:
            return None

    # ── Historical price access ────────────────────────────────

    @staticmethod
    def get_price_history(
        token_id: str,
        max_snapshots: int = 60,
    ) -> list[dict]:
        """Get recent price snapshots for a token.

        Returns the most recent snapshots from the WebSocket feed cache.
        Each snapshot includes timestamp, mid, bid, ask.

        Args:
            token_id: The CLOB token ID.
            max_snapshots: Maximum number of snapshots to return.

        Returns:
            List of price snapshots (newest first), each:
            {"timestamp": str, "mid": float, "bid": float, "ask": float}
            Empty list if no history available.
        """
        try:
            from services.ws_feeds import get_feed_manager

            fm = get_feed_manager()
            if fm and hasattr(fm.cache, "get_price_history"):
                return fm.cache.get_price_history(token_id, max_snapshots=max_snapshots)
        except Exception:
            pass
        return []

    @staticmethod
    def get_price_change(
        token_id: str,
        lookback_seconds: int = 300,
    ) -> Optional[dict]:
        """Get price change over a lookback period.

        Args:
            token_id: The CLOB token ID.
            lookback_seconds: How far back to look (default 5 minutes).

        Returns:
            Dict with price change info, or None:
            {
                "current_mid": float,
                "prior_mid": float,
                "change_abs": float,
                "change_pct": float,
                "snapshots_in_window": int,
            }
        """
        try:
            from services.ws_feeds import get_feed_manager

            fm = get_feed_manager()
            if fm and hasattr(fm.cache, "get_price_change"):
                return fm.cache.get_price_change(token_id, lookback_seconds=lookback_seconds)
        except Exception:
            pass
        return None

    # ── News data access ───────────────────────────────────────

    @staticmethod
    def get_recent_news(
        query: str = "",
        max_articles: int = 20,
    ) -> list[dict]:
        """Get recent news articles, optionally filtered by query.

        Args:
            query: Optional search query to filter articles.
            max_articles: Maximum number of articles to return.

        Returns:
            List of article dicts with title, source, published_at, summary.
        """
        try:
            from services.news.feed_service import news_feed_service

            articles = news_feed_service.search_articles(query=query, limit=max_articles)
            return [
                {
                    "title": getattr(a, "title", ""),
                    "source": getattr(a, "source", ""),
                    "published_at": str(getattr(a, "published_at", "")),
                    "summary": getattr(a, "summary", getattr(a, "description", "")),
                    "url": getattr(a, "url", getattr(a, "link", "")),
                }
                for a in (articles or [])
            ]
        except Exception:
            return []

    @staticmethod
    def get_news_for_market(
        market: Any,
        max_articles: int = 10,
    ) -> list[dict]:
        """Get news articles semantically matched to a specific market.

        Args:
            market: A Market object.
            max_articles: Maximum number of articles to return.

        Returns:
            List of article dicts with relevance_score.
        """
        try:
            from services.news.semantic_matcher import semantic_matcher

            question = getattr(market, "question", "")
            if not question:
                return []

            matches = semantic_matcher.find_matches(question, top_k=max_articles)
            return [
                {
                    "title": m.get("title", ""),
                    "source": m.get("source", ""),
                    "relevance_score": m.get("score", 0),
                    "published_at": str(m.get("published_at", "")),
                    "summary": m.get("summary", ""),
                }
                for m in (matches or [])
            ]
        except Exception:
            return []

    # ── Data source access ───────────────────────────────

    @staticmethod
    async def get_data_records(
        source_slug: str | None = None,
        source_slugs: list[str] | None = None,
        limit: int = 200,
        geotagged: bool | None = None,
        category: str | None = None,
        since: str | None = None,
    ) -> list[dict]:
        """Read normalized records from the data source store."""
        try:
            from services.data_source_sdk import DataSourceSDK

            return await DataSourceSDK.get_records(
                source_slug=source_slug,
                source_slugs=source_slugs,
                limit=limit,
                geotagged=geotagged,
                category=category,
                since=since,
            )
        except Exception as e:
            logger.warning("StrategySDK.get_data_records failed: %s", e)
            return []

    @staticmethod
    async def get_latest_data_record(
        source_slug: str,
        external_id: str | None = None,
    ) -> dict | None:
        """Read the newest normalized record for a source (optionally by external_id)."""
        try:
            from services.data_source_sdk import DataSourceSDK

            return await DataSourceSDK.get_latest_record(
                source_slug=source_slug,
                external_id=external_id,
            )
        except Exception as e:
            logger.warning("StrategySDK.get_latest_data_record failed: %s", e)
            return None

    @staticmethod
    async def run_data_source(source_slug: str, max_records: int = 500) -> dict:
        """Trigger a source run and return ingestion summary."""
        try:
            from services.data_source_sdk import DataSourceSDK

            return await DataSourceSDK.run_source(source_slug=source_slug, max_records=max_records)
        except Exception as e:
            logger.warning("StrategySDK.run_data_source failed: %s", e)
            return {}

    @staticmethod
    async def list_data_sources(
        enabled_only: bool = True,
        source_key: str | None = None,
        include_code: bool = False,
    ) -> list[dict]:
        """List DB-backed data sources."""
        try:
            from services.data_source_sdk import DataSourceSDK

            return await DataSourceSDK.list_sources(
                enabled_only=enabled_only,
                source_key=source_key,
                include_code=include_code,
            )
        except Exception as e:
            logger.warning("StrategySDK.list_data_sources failed: %s", e)
            return []

    @staticmethod
    async def get_data_source(source_slug: str, include_code: bool = True) -> dict:
        """Fetch one DB-backed data source by slug."""
        try:
            from services.data_source_sdk import DataSourceSDK

            return await DataSourceSDK.get_source(source_slug=source_slug, include_code=include_code)
        except Exception as e:
            logger.warning("StrategySDK.get_data_source failed: %s", e)
            return {}

    @staticmethod
    def validate_data_source(source_code: str, class_name: str | None = None) -> dict:
        """Validate source code before create/update."""
        try:
            from services.data_source_sdk import DataSourceSDK

            return DataSourceSDK.validate_source(source_code=source_code, class_name=class_name)
        except Exception as e:
            logger.warning("StrategySDK.validate_data_source failed: %s", e)
            return {
                "valid": False,
                "errors": [str(e)],
                "warnings": [],
                "class_name": None,
                "source_name": None,
                "source_description": None,
                "capabilities": {"has_fetch": False, "has_fetch_async": False, "has_transform": False},
            }

    @staticmethod
    async def create_data_source(
        *,
        slug: str,
        source_code: str,
        source_key: str = "custom",
        source_kind: str = "python",
        name: str | None = None,
        description: str | None = None,
        class_name: str | None = None,
        config: dict[str, Any] | None = None,
        config_schema: dict[str, Any] | None = None,
        enabled: bool = True,
        is_system: bool = False,
        sort_order: int = 0,
    ) -> dict:
        """Create a new DB-backed data source."""
        try:
            from services.data_source_sdk import DataSourceSDK

            return await DataSourceSDK.create_source(
                slug=slug,
                source_code=source_code,
                source_key=source_key,
                source_kind=source_kind,
                name=name,
                description=description,
                class_name=class_name,
                config=config,
                config_schema=config_schema,
                enabled=enabled,
                is_system=is_system,
                sort_order=sort_order,
            )
        except Exception as e:
            logger.warning("StrategySDK.create_data_source failed: %s", e)
            return {}

    @staticmethod
    async def update_data_source(
        source_slug: str,
        *,
        slug: str | None = None,
        source_key: str | None = None,
        source_kind: str | None = None,
        name: str | None = None,
        description: str | None = None,
        source_code: str | None = None,
        class_name: str | None = None,
        config: dict[str, Any] | None = None,
        config_schema: dict[str, Any] | None = None,
        enabled: bool | None = None,
        unlock_system: bool = False,
    ) -> dict:
        """Update an existing DB-backed data source."""
        try:
            from services.data_source_sdk import DataSourceSDK

            return await DataSourceSDK.update_source(
                source_slug=source_slug,
                slug=slug,
                source_key=source_key,
                source_kind=source_kind,
                name=name,
                description=description,
                source_code=source_code,
                class_name=class_name,
                config=config,
                config_schema=config_schema,
                enabled=enabled,
                unlock_system=unlock_system,
            )
        except Exception as e:
            logger.warning("StrategySDK.update_data_source failed: %s", e)
            return {}

    @staticmethod
    async def delete_data_source(
        source_slug: str,
        *,
        tombstone_system_source: bool = True,
        unlock_system: bool = False,
        reason: str = "deleted_via_strategy_sdk",
    ) -> dict:
        """Delete a data source by slug."""
        try:
            from services.data_source_sdk import DataSourceSDK

            return await DataSourceSDK.delete_source(
                source_slug=source_slug,
                tombstone_system_source=tombstone_system_source,
                unlock_system=unlock_system,
                reason=reason,
            )
        except Exception as e:
            logger.warning("StrategySDK.delete_data_source failed: %s", e)
            return {}

    @staticmethod
    async def reload_data_source(source_slug: str) -> dict:
        """Reload a data source runtime by slug."""
        try:
            from services.data_source_sdk import DataSourceSDK

            return await DataSourceSDK.reload_source(source_slug=source_slug)
        except Exception as e:
            logger.warning("StrategySDK.reload_data_source failed: %s", e)
            return {}

    @staticmethod
    async def get_data_source_runs(source_slug: str, limit: int = 20) -> list[dict]:
        """Read recent run history for a source."""
        try:
            from services.data_source_sdk import DataSourceSDK

            return await DataSourceSDK.get_recent_runs(source_slug=source_slug, limit=limit)
        except Exception as e:
            logger.warning("StrategySDK.get_data_source_runs failed: %s", e)
            return []
