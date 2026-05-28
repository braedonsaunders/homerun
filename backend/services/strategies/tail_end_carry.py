"""Tail-end carry opportunity filter.

Portable pattern adapted from public tail-end Polymarket bots: filter for
high-probability outcomes close to resolution, then only keep entries with
liquidity/spread quality and non-trivial expected repricing room.
"""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass
from collections import deque
from datetime import datetime, timezone
from typing import Any, Optional

from config import settings
from models import Opportunity, Event, Market
from models.opportunity import MispricingType
from services.strategies.base import (
    BaseStrategy,
    DecisionCheck,
    StrategyDecision,
    ExitDecision,
    ScoringWeights,
    SizingConfig,
    _trader_size_limits,
    make_aware,
    utcnow,
)
import logging

from utils.converters import to_float, to_confidence, clamp
from utils.signal_helpers import signal_payload, days_to_resolution, selected_probability
from utils.converters import safe_float
from services.quality_filter import QualityFilterOverrides

logger = logging.getLogger(__name__)


@dataclass
class _StopLossPosition:
    config: dict[str, Any]
    entry_price: float
    age_minutes: float | None
    highest_price: float | None
    outcome_idx: int
    strategy_context: dict[str, Any]

# -- Market category constants ------------------------------------------------

CATEGORY_SPORTS = "sports"
CATEGORY_ESPORTS = "esports"
CATEGORY_CRYPTO = "crypto"
CATEGORY_POLITICAL = "political"
CATEGORY_OTHER = "other"

_SPORTS_KEYWORDS = [
    "exact score", "correct score",
    "vs.", "vs ", " fc ", "hotspur", "united", "rovers", "city fc", "atletico",
    "juventus", "barcelona", "bayern", "borussia", "schalke", "heracles",
    "almelo", "napoli", "lazio", "inter ", "milan", "fiorentina", "roma ",
    "o/u ", "spread:", "moneyline", "hatters", "governors", "warriors",
    "broncos", "flashes", "mavericks", "celtics", "nets vs", "heat vs",
    "tigers", "bulldogs", "wildcats", "hawks", "eagles", "bears", "lions",
    "panthers", "knights", "raiders", "saints", "cardinals", "rams",
    "nba", "nfl", "mlb", "nhl", "cbb", "ncaa", "premier league",
    "la liga", "serie a", "bundesliga", "ligue 1", "eredivisie",
    "champions league", "europa league", "copa ", "primera division",
    " sc ", " afc ", "real madrid", "psg", "monaco", "tottenham",
    "crystal palace", "hull city", "holstein",
]

_ESPORTS_KEYWORDS = [
    "lol:", "league of legends", "counter-strike", "esports", "rift legends",
    "dota", "valorant", "overwatch", "cs2", "cs:", "esl pro",
    "game 1 winner", "game 2 winner", "game 3 winner",
    "(bo3)", "(bo5)", "(bo1)",
]

_CRYPTO_KEYWORDS = [
    "bitcoin", "btc", "ethereum", "eth", "solana", "sol ", "xrp",
    "crypto", "up or down", "above $", "below $", "price of ",
]

_POLITICAL_KEYWORDS = [
    "election", "vote", "primary", "congress", "senate", "president",
    "prime minister", "parliament", "seats", "ballot", "referendum",
    "trump", "biden", "paxton", "governor", "mayor", "iran",
    "us forces", "strike ", "war ", "military", "geopolitical",
    "nepal", "pakistan", "ceasefire", "tariff",
]


def _classify_market(text: str, sports_market_type: str | None) -> str:
    """Classify a market into a category based on text content and metadata."""
    if sports_market_type and sports_market_type in ("moneyline", "totals", "spreads", "props"):
        return CATEGORY_SPORTS
    lower = text.lower()
    for kw in _ESPORTS_KEYWORDS:
        if kw in lower:
            return CATEGORY_ESPORTS
    for kw in _SPORTS_KEYWORDS:
        if kw in lower:
            return CATEGORY_SPORTS
    for kw in _CRYPTO_KEYWORDS:
        if kw in lower:
            return CATEGORY_CRYPTO
    for kw in _POLITICAL_KEYWORDS:
        if kw in lower:
            return CATEGORY_POLITICAL
    return CATEGORY_OTHER


def _parse_game_start_time(raw: Any) -> datetime | None:
    if raw is None:
        return None
    if isinstance(raw, datetime):
        if raw.tzinfo is None:
            return raw.replace(tzinfo=timezone.utc)
        return raw
    s = str(raw).strip()
    if not s:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S%z", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(s, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    return None


def _is_bool_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


class TailEndCarryStrategy(BaseStrategy):
    """Find near-resolution high-probability outcomes with executable carry."""

    strategy_type = "tail_end_carry"
    name = "Tail-End Carry"
    description = "Near-expiry high-probability carry with liquidity and spread gates"
    mispricing_type = "within_market"
    requires_resolution_date = True
    subscriptions = ["market_data_refresh"]

    quality_filter_overrides = QualityFilterOverrides(
        min_roi=1.0,
        min_position_size=20.0,
        min_absolute_profit=3.0,
        max_resolution_months=1.0,
    )

    pipeline_defaults = {
        "min_edge_percent": 1.0,
        "min_confidence": 0.35,
        "max_risk_score": 0.78,
    }

    # Composable evaluate pipeline: score = edge*2.5 + conf*28 + entry*6 - risk*9 + time_bonus
    scoring_weights = ScoringWeights(
        edge_weight=2.5,
        confidence_weight=28.0,
        risk_penalty=9.0,
    )
    # Sizing uses adaptive policy -- override compute_size in evaluate
    sizing_config = SizingConfig()

    default_config = {
        "min_probability": 0.85,
        "max_probability": 0.905,
        "min_upside_percent": 10.0,
        # Sports-specific entry overrides
        "sports_min_probability": 0.90,
        "sports_max_days_to_resolution": 0.25,
        "min_days_to_resolution": 0.0,
        "max_days_to_resolution": 1.0,
        "min_liquidity": 1500.0,
        "max_spread": 0.05,
        "min_repricing_buffer": 0.015,
        "repricing_weight": 0.45,
        "exclude_market_keywords": [
            "exact score", "correct score",
            "lol:", "counter-strike",
            "tweets", "league of legends", "esports", "rift legends",
            "dota", "valorant", "cs2", "cs:", "esl pro",
        ],
        # Spread market gate — sports spreads swing violently mid-game
        # and the trailing stop locks in losses on temporary adverse scores.
        # Block them entirely; moneyline / O-U totals are safer tail-carry targets.
        "block_spread_markets": True,
        "panic_drop_threshold": 0.08,
        "panic_window_points": 6,
        "panic_recovery_ratio_max": 0.20,
        "take_profit_pct": 10.0,
        "smart_take_profit_enabled": True,
        "smart_take_profit_min_pnl_pct": 10.0,
        "smart_take_profit_max_price_headroom": 0.03,
        "inversion_stop_enabled": True,
        "inversion_price_threshold": 0.50,
        "trailing_stop_enabled": True,
        "trailing_stop_pct": 12.0,
        "sports_inversion_stop_enabled": False,
        "sports_trailing_stop_pct": 30.0,
        # ----------------------------------------------------------------
        # Time-weighted exit shaping (#1, #2, #3)
        # ----------------------------------------------------------------
        # Tightens trailing-stop pct + raises inversion floor as TTR shrinks.
        # Each row: [minutes_left_at_or_below, trailing_stop_pct, inversion_floor].
        # The first row whose minutes_left_at_or_below >= remaining_minutes wins.
        # Trailing list is independent from sports; sports gets a looser variant.
        "time_weighted_stop_enabled": True,
        "time_weighted_stop_schedule": [
            [30,    2.0,  0.85],
            [120,   5.0,  0.72],
            [360,  10.0,  0.60],
            [1440, 18.0,  0.50],
            [99999, 30.0, 0.40],
        ],
        "sports_time_weighted_stop_schedule": [
            [30,    5.0,  0.70],
            [120,  10.0,  0.55],
            [360,  20.0,  0.45],
            [1440, 35.0,  0.35],
            [99999, 50.0, 0.30],
        ],
        # Resolution-window scale-out ladder. Each tier evaluated in order;
        # first matching unhit tier fires. ``tighten_trailing_floor_pct`` is
        # a synthetic ratchet so the live lifecycle (which currently no-ops
        # ``reduce``) still locks in profit on the next tick.
        "scale_out_enabled": True,
        "scale_out_tiers": [
            {"minutes_left_max": 120, "min_price": 0.95, "exit_fraction": 0.25, "tighten_trailing_floor_pct": 4.0},
            {"minutes_left_max": 30,  "min_price": 0.97, "exit_fraction": 0.50, "tighten_trailing_floor_pct": 2.0},
            {"minutes_left_max": 5,   "min_price": 0.0,  "exit_fraction": 1.0,  "tighten_trailing_floor_pct": 0.0},
        ],
        # Velocity stop: when within ``velocity_active_minutes_left`` of
        # resolution, exit immediately if price drops at least
        # ``velocity_drop_threshold_pct`` over a rolling
        # ``velocity_window_seconds`` window. Catches the cliff-drops that
        # absolute-drawdown stops miss because they happen too fast.
        "velocity_stop_enabled": True,
        "velocity_window_seconds": 300.0,
        "velocity_drop_threshold_pct": 3.0,
        "velocity_active_minutes_left": 60.0,
        "velocity_min_samples": 3,
        "sports_sizing_multiplier": 0.45,
        "skip_live_games": True,
        "live_game_buffer_minutes": 15.0,
        "session_timeout_seconds": 180,
        "resolution_hold_enabled": True,
        "resolution_hold_minutes": 360.0,
        "sports_resolution_hold_minutes": 150.0,
        "resolution_hold_max_loss_pct": 25.0,
        # Pre-resolution structural exit (gap cap).  Carrying a ~0.9 favorite
        # all the way INTO resolution exposes the full notional to a binary
        # gap (0.9 -> 0 in one step, no price for a stop to catch).  When > 0,
        # force a CLOSE while still tradable once the market is within this
        # many minutes of resolution — capturing ~all of the carry while
        # NEVER holding through the resolution instant.  0 = disabled (pure
        # carry-to-resolution; default, to preserve existing edge until the
        # operator opts in).  Sports markets use the sports value when set.
        "pre_resolution_force_exit_minutes": 0.0,
        "sports_pre_resolution_force_exit_minutes": 0.0,
        # Exit-trigger mark basis.  "liquidation_vwap" (default) prices
        # stops/take-profits off the realizable bid-side VWAP to clear the
        # position's shares — what we could ACTUALLY exit at — instead of
        # the optimistic mid.  Set to "mid" to revert to midpoint marking.
        "exit_mark_mode": "liquidation_vwap",
        "max_hold_minutes": 1440.0,
        "price_policy": "taker_limit",
        "time_in_force": "IOC",
        "allow_taker_limit_buy_above_signal": True,
        "aggressive_limit_buy_submit_as_gtc": False,
        "immediate_break_even_stop_enabled": True,
        "immediate_break_even_stop_buffer_pct": 0.5,
        "max_market_data_age_ms": 15000,
        "require_strict_ws_pricing": True,
        "strict_ws_price_sources": ["ws_strict", "redis_strict"],
    }

    def __init__(self) -> None:
        super().__init__()
        self.config = dict(self.default_config)
        self.min_profit = 0.01

    @staticmethod
    def _book_value(payload: Optional[dict], key: str) -> Optional[float]:
        if not isinstance(payload, dict):
            return None
        val = payload.get(key)
        if isinstance(val, (int, float)):
            return float(val)
        return None

    def _extract_side_book(
        self,
        market: Market,
        prices: dict[str, dict],
        outcome: str,
    ) -> tuple[float, Optional[float], Optional[float], Optional[str]]:
        yes = safe_float(getattr(market, "yes_price", 0.0))
        no = safe_float(getattr(market, "no_price", 0.0))
        tokens = list(getattr(market, "clob_token_ids", []) or [])

        if outcome == "YES":
            token_id = tokens[0] if len(tokens) > 0 else None
            payload = prices.get(token_id) if token_id else None
            mid = self._book_value(payload, "mid")
            if mid is None:
                mid = self._book_value(payload, "price")
            price = mid if mid is not None else yes
            bid = self._book_value(payload, "bid") or self._book_value(payload, "best_bid")
            ask = self._book_value(payload, "ask") or self._book_value(payload, "best_ask")
            return price, bid, ask, token_id

        token_id = tokens[1] if len(tokens) > 1 else None
        payload = prices.get(token_id) if token_id else None
        mid = self._book_value(payload, "mid")
        if mid is None:
            mid = self._book_value(payload, "price")
        price = mid if mid is not None else no
        bid = self._book_value(payload, "bid") or self._book_value(payload, "best_bid")
        ask = self._book_value(payload, "ask") or self._book_value(payload, "best_ask")
        return price, bid, ask, token_id

    # ------------------------------------------------------------------
    # Keyword exclusion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _append_text(chunks: list[str], value: Any) -> None:
        text = str(value or "").strip().lower()
        if text:
            chunks.append(text)

    @staticmethod
    def _normalize_excluded_keywords(value: Any) -> list[str]:
        if isinstance(value, str):
            candidates: list[Any] = [token.strip() for token in value.split(",")]
        elif isinstance(value, list):
            candidates = list(value)
        else:
            candidates = []
        out: list[str] = []
        seen: set[str] = set()
        for raw in candidates:
            token = str(raw or "").strip().lower()
            if not token or token in seen:
                continue
            seen.add(token)
            out.append(token)
        return out

    @staticmethod
    def _keyword_in_text(keyword: str, text: str) -> bool:
        if not keyword or not text:
            return False
        if len(keyword) <= 4 and keyword.replace("-", "").replace("_", "").isalnum():
            return re.search(rf"\b{re.escape(keyword)}\b", text) is not None
        return keyword in text

    @classmethod
    def _first_blocked_keyword(cls, text: str, excluded_keywords: list[str]) -> str | None:
        for keyword in excluded_keywords:
            if cls._keyword_in_text(keyword, text):
                return keyword
        return None

    @classmethod
    def _market_text(cls, market: Market, event: Optional[Event]) -> str:
        chunks: list[str] = []
        for value in (market.id, market.question, market.slug, market.group_item_title, market.event_slug):
            cls._append_text(chunks, value)
        for tag in list(getattr(market, "tags", []) or []):
            cls._append_text(chunks, tag)
        if event is not None:
            for value in (event.id, event.slug, event.title, event.category):
                cls._append_text(chunks, value)
            for tag in list(getattr(event, "tags", []) or []):
                cls._append_text(chunks, tag)
        return " | ".join(chunks)

    @classmethod
    def _signal_market_text(cls, signal: Any, payload: dict[str, Any]) -> str:
        chunks: list[str] = []
        for value in (
            getattr(signal, "market_question", None),
            payload.get("market_question"),
            payload.get("title"),
            payload.get("description"),
            payload.get("market_id"),
            payload.get("market_slug"),
            payload.get("event_title"),
            payload.get("event_slug"),
            payload.get("category"),
        ):
            cls._append_text(chunks, value)
        markets = payload.get("markets")
        if isinstance(markets, list):
            for raw_market in markets[:2]:
                if not isinstance(raw_market, dict):
                    continue
                for value in (
                    raw_market.get("id"),
                    raw_market.get("question"),
                    raw_market.get("slug"),
                    raw_market.get("event_slug"),
                    raw_market.get("group_item_title"),
                ):
                    cls._append_text(chunks, value)
                tags = raw_market.get("tags")
                if isinstance(tags, list):
                    for tag in tags:
                        cls._append_text(chunks, tag)
                else:
                    cls._append_text(chunks, tags)
        event = payload.get("event")
        if isinstance(event, dict):
            for value in (event.get("id"), event.get("slug"), event.get("title"), event.get("category")):
                cls._append_text(chunks, value)
            tags = event.get("tags")
            if isinstance(tags, list):
                for tag in tags:
                    cls._append_text(chunks, tag)
            else:
                cls._append_text(chunks, tags)
        return " | ".join(chunks)

    # ------------------------------------------------------------------
    # Market category helpers (Fix #1)
    # ------------------------------------------------------------------

    @classmethod
    def _classify_market_from_model(cls, market: Market, event: Optional[Event]) -> str:
        """Classify a Market model object at detect time."""
        sports_type = getattr(market, "sports_market_type", None)
        text = cls._market_text(market, event)
        return _classify_market(text, sports_type)

    @classmethod
    def _classify_market_from_payload(cls, payload: dict[str, Any]) -> str:
        """Classify from signal payload at evaluate/exit time."""
        markets = payload.get("markets")
        sports_type = None
        if isinstance(markets, list) and markets:
            m0 = markets[0] if isinstance(markets[0], dict) else {}
            sports_type = m0.get("sports_market_type")
        text_parts: list[str] = []
        cls._append_text(text_parts, payload.get("title"))
        cls._append_text(text_parts, payload.get("description"))
        cls._append_text(text_parts, payload.get("category"))
        cls._append_text(text_parts, payload.get("event_title"))
        if isinstance(markets, list):
            for m in markets[:2]:
                if isinstance(m, dict):
                    cls._append_text(text_parts, m.get("question"))
                    cls._append_text(text_parts, m.get("slug"))
                    cls._append_text(text_parts, m.get("sports_market_type"))
                    cls._append_text(text_parts, m.get("group_item_title"))
        return _classify_market(" | ".join(text_parts), sports_type)

    @classmethod
    def _classify_from_exit_config(cls, config: dict, strategy_context: Optional[dict[str, Any]] = None) -> str:
        """Classify from the strategy_exit_config stored on the order."""
        cached = config.get("_market_category") or config.get("market_category")
        if cached and isinstance(cached, str):
            return cached

        context = strategy_context if isinstance(strategy_context, dict) else {}
        context_cached = context.get("market_category")
        if isinstance(context_cached, str) and context_cached.strip():
            return context_cached.strip().lower()

        text_parts: list[str] = []
        cls._append_text(text_parts, config.get("_market_question"))
        cls._append_text(text_parts, config.get("market_question"))
        cls._append_text(text_parts, config.get("_market_slug"))
        cls._append_text(text_parts, config.get("market_slug"))
        cls._append_text(text_parts, config.get("_event_title"))
        cls._append_text(text_parts, config.get("event_title"))
        cls._append_text(text_parts, config.get("_category"))
        cls._append_text(text_parts, context.get("market_question"))
        cls._append_text(text_parts, context.get("event_title"))
        sports_type = config.get("_sports_market_type") or config.get("sports_market_type")
        if not sports_type:
            sports_type = context.get("sports_market_type")
        return _classify_market(" | ".join(text_parts), sports_type)

    # ------------------------------------------------------------------
    # Spread market detection
    # ------------------------------------------------------------------

    _SPREAD_PATTERNS = re.compile(
        r"(?:^|\b)spread\s*:", re.IGNORECASE,
    )

    @classmethod
    def _is_spread_market(cls, text: str) -> bool:
        """Detect spread-type sports markets (e.g. 'Spread: Team X (-1.5)')."""
        return cls._SPREAD_PATTERNS.search(text) is not None

    # ------------------------------------------------------------------
    # Live game detection (Fix #3)
    # ------------------------------------------------------------------

    @staticmethod
    def _is_live_game(market: Market, now: datetime, buffer_minutes: float) -> bool:
        """Check if a sports/esports market's game is currently live or about to start."""
        raw_start = getattr(market, "game_start_time", None)
        game_start = _parse_game_start_time(raw_start)
        if game_start is None:
            return False
        buffer_seconds = buffer_minutes * 60.0
        return (game_start.timestamp() - buffer_seconds) <= now.timestamp()

    @staticmethod
    def _is_live_game_from_payload(payload: dict, now: datetime, buffer_minutes: float) -> bool:
        """Check if market is live from signal payload."""
        markets = payload.get("markets")
        if not isinstance(markets, list) or not markets:
            return False
        m0 = markets[0] if isinstance(markets[0], dict) else {}
        game_start = _parse_game_start_time(m0.get("game_start_time"))
        if game_start is None:
            return False
        buffer_seconds = buffer_minutes * 60.0
        return (game_start.timestamp() - buffer_seconds) <= now.timestamp()

    def _tail_end_limits(self, params: dict[str, Any]) -> dict[str, Any]:
        cfg = dict(self.default_config)
        if params:
            cfg.update(params)

        min_probability = clamp(safe_float(cfg.get("min_probability"), 0.85), 0.5, 0.995)
        max_probability = clamp(safe_float(cfg.get("max_probability"), 0.999), min_probability, 0.999)
        sports_min_probability = clamp(safe_float(cfg.get("sports_min_probability"), 0.90), 0.5, 0.995)
        min_days = max(0.0, safe_float(cfg.get("min_days_to_resolution"), 0.0))
        max_days = max(min_days + 0.005, safe_float(cfg.get("max_days_to_resolution"), 1.0))
        sports_max_days = max(min_days + 0.005, safe_float(cfg.get("sports_max_days_to_resolution"), 0.25))
        min_liquidity = max(100.0, safe_float(cfg.get("min_liquidity"), 1000.0))
        max_spread = clamp(safe_float(cfg.get("max_spread"), 0.05), 0.005, 0.20)
        min_upside_percent = clamp(safe_float(cfg.get("min_upside_percent"), 5.0), 5.0, 100.0)
        min_repricing_buffer = clamp(safe_float(cfg.get("min_repricing_buffer"), 0.015), 0.005, 0.10)
        repricing_weight = clamp(safe_float(cfg.get("repricing_weight"), 0.45), 0.10, 0.90)
        panic_drop_threshold = clamp(safe_float(cfg.get("panic_drop_threshold"), 0.08), 0.02, 0.30)
        panic_window_points = max(3, int(safe_float(cfg.get("panic_window_points"), 6)))
        panic_recovery_ratio_max = clamp(safe_float(cfg.get("panic_recovery_ratio_max"), 0.20), 0.0, 0.9)
        block_spread_markets = _is_bool_true(cfg.get("block_spread_markets", True))
        skip_live_games = _is_bool_true(cfg.get("skip_live_games", True))
        live_game_buffer_minutes = max(0.0, safe_float(cfg.get("live_game_buffer_minutes"), 15.0))
        allow_taker_limit_buy_above_signal = _is_bool_true(cfg.get("allow_taker_limit_buy_above_signal", True))
        aggressive_limit_buy_submit_as_gtc = _is_bool_true(cfg.get("aggressive_limit_buy_submit_as_gtc", False))

        return {
            "min_probability": min_probability,
            "max_probability": max_probability,
            "sports_min_probability": sports_min_probability,
            "sports_max_days_to_resolution": sports_max_days,
            "min_days_to_resolution": min_days,
            "max_days_to_resolution": max_days,
            "min_liquidity": min_liquidity,
            "min_upside_percent": min_upside_percent,
            "max_spread": max_spread,
            "min_repricing_buffer": min_repricing_buffer,
            "repricing_weight": repricing_weight,
            "panic_drop_threshold": panic_drop_threshold,
            "panic_window_points": panic_window_points,
            "panic_recovery_ratio_max": panic_recovery_ratio_max,
            "block_spread_markets": block_spread_markets,
            "exclude_market_keywords": self._normalize_excluded_keywords(cfg.get("exclude_market_keywords")),
            "skip_live_games": skip_live_games,
            "live_game_buffer_minutes": live_game_buffer_minutes,
            "allow_taker_limit_buy_above_signal": allow_taker_limit_buy_above_signal,
            "aggressive_limit_buy_submit_as_gtc": aggressive_limit_buy_submit_as_gtc,
            "price_policy": str(cfg.get("price_policy", "taker_limit") or "taker_limit"),
            "time_in_force": str(cfg.get("time_in_force", "IOC") or "IOC"),
        }

    def _tail_end_checks(
        self,
        *,
        source: str,
        strategy_type: str,
        signal_text: str,
        entry_price: float,
        dtr: float | None,
        category: str,
        blocked_keyword: str | None,
        is_spread: bool,
        liquidity: float,
        observed_spread: float,
        panic_guard_ok: bool,
        is_live: bool,
        limits: dict[str, Any],
    ) -> list[DecisionCheck]:
        effective_min_entry = (
            limits["sports_min_probability"] if category in (CATEGORY_SPORTS, CATEGORY_ESPORTS) else limits["min_probability"]
        )
        effective_max_days = (
            limits["sports_max_days_to_resolution"]
            if category in (CATEGORY_SPORTS, CATEGORY_ESPORTS)
            else limits["max_days_to_resolution"]
        )
        effective_days_ok = dtr is not None and limits["min_days_to_resolution"] <= dtr <= effective_max_days
        upside_pct = ((1.0 - entry_price) / entry_price * 100.0) if entry_price > 0.0 else 0.0
        upside_ok = upside_pct >= float(limits.get("min_upside_percent", 5.0))
        spread_ok = observed_spread <= float(limits["max_spread"])
        live_game_ok = not is_live

        checks = [
            DecisionCheck("source", "Scanner source", source == "scanner", detail="Requires source=scanner."),
            DecisionCheck("strategy", "Tail carry strategy type", strategy_type == "tail_end_carry", detail="strategy=tail_end_carry"),
            DecisionCheck(
                "spread_market",
                "Not a spread market",
                not is_spread,
                detail="Spread markets blocked (volatile mid-game swings)" if is_spread else "ok",
            ),
            DecisionCheck(
                "keyword_block",
                "Market keyword allowed",
                blocked_keyword is None,
                detail=f"blocked keyword={blocked_keyword}" if blocked_keyword else "ok",
            ),
            DecisionCheck(
                "liquidity_floor",
                "Hard liquidity floor",
                liquidity >= float(limits["min_liquidity"]),
                score=liquidity,
                detail=f">= {float(limits['min_liquidity']):.0f}",
            ),
            DecisionCheck(
                "entry",
                "Entry probability band",
                effective_min_entry <= entry_price <= float(limits["max_probability"]),
                score=entry_price,
                detail=f"[{effective_min_entry:.3f}, {float(limits['max_probability']):.3f}]{' (sports override)' if category in (CATEGORY_SPORTS, CATEGORY_ESPORTS) else ''}",
            ),
            DecisionCheck(
                "resolution_window",
                "Resolution window",
                effective_days_ok,
                score=dtr,
                detail=f"[{float(limits['min_days_to_resolution']):.2f}, {effective_max_days:.2f}] days{' (sports override)' if category in (CATEGORY_SPORTS, CATEGORY_ESPORTS) else ''}",
            ),
            DecisionCheck(
                "upside",
                "Max settlement upside floor",
                upside_ok,
                score=upside_pct,
                detail=f">= {float(limits.get('min_upside_percent', 5.0)):.2f}%",
            ),
            DecisionCheck(
                "book_spread",
                "Bid/ask spread within limit",
                spread_ok,
                score=observed_spread,
                detail=f"<= {float(limits['max_spread']):.3f}",
            ),
            DecisionCheck(
                "panic_guard",
                "Recent price action not in panic unwind",
                panic_guard_ok,
                detail="panic/recovery guard",
            ),
        ]

        if limits["skip_live_games"]:
            checks.append(
                DecisionCheck(
                    "live_game",
                    "Not a live game",
                    live_game_ok,
                    detail=f"category={category}, live={is_live}",
                ),
            )

        return checks

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect(
        self,
        events: list[Event],
        markets: list[Market],
        prices: dict[str, dict],
    ) -> list[Opportunity]:
        cfg = dict(self.default_config)
        cfg.update(getattr(self, "config", {}) or {})

        limits = self._tail_end_limits(cfg)
        min_probability = float(limits["min_probability"])
        min_upside_percent = float(limits["min_upside_percent"])
        sports_min_probability = float(limits["sports_min_probability"])
        sports_max_days = float(limits["sports_max_days_to_resolution"])
        min_days = float(limits["min_days_to_resolution"])
        max_days = float(limits["max_days_to_resolution"])
        min_liquidity = float(limits["min_liquidity"])
        max_spread = float(limits["max_spread"])
        min_repricing_buffer = float(limits["min_repricing_buffer"])
        repricing_weight = float(limits["repricing_weight"])
        panic_drop_threshold = float(limits["panic_drop_threshold"])
        panic_window_points = int(limits["panic_window_points"])
        panic_recovery_ratio_max = float(limits["panic_recovery_ratio_max"])
        excluded_keywords = self._normalize_excluded_keywords(cfg.get("exclude_market_keywords"))
        block_spread_markets = bool(limits["block_spread_markets"])
        skip_live_games = bool(limits["skip_live_games"])
        live_game_buffer_minutes = float(limits["live_game_buffer_minutes"])
        allow_taker_limit_buy_above_signal = bool(limits["allow_taker_limit_buy_above_signal"])

        event_by_market: dict[str, Event] = {}
        for event in events:
            for event_market in event.markets:
                event_by_market[event_market.id] = event

        now = utcnow().astimezone(timezone.utc)
        candidates: list[tuple[float, Opportunity]] = []
        history_by_key = self.state.setdefault("tail_carry_price_history", {})
        raw_detected_count = 0

        for market in markets:
            if market.closed or not market.active:
                continue
            event_for_market = event_by_market.get(market.id)
            market_text = self._market_text(market, event_for_market)
            if (
                len(list(getattr(market, "clob_token_ids", []) or [])) < 2
                and len(list(getattr(market, "outcome_prices", []) or [])) < 2
            ):
                continue

            end_date = make_aware(getattr(market, "end_date", None))
            if end_date is None:
                continue

            # For grouped/multi-outcome markets, the event-level end_date
            # may be the earliest sub-market (e.g. "March 31") while the
            # specific sub-market resolves much later (e.g. "December 31,
            # 2026").  Extract the actual date from group_item_title if it
            # parses to a LATER date, and use that for the resolution check.
            group_title = str(getattr(market, "group_item_title", "") or "").strip()
            if group_title:
                # Parse common date formats from group_item_title using
                # stdlib only (third-party date parsers are blocked by sandbox).
                # Handles: "March 31, 2026", "December 31, 2026", "2026-12-31"
                import re as _re
                _MONTHS = {
                    "january": 1, "february": 2, "march": 3, "april": 4,
                    "may": 5, "june": 6, "july": 7, "august": 8,
                    "september": 9, "october": 10, "november": 11, "december": 12,
                }
                _parsed_group_date = None
                # Try "Month Day, Year" pattern
                _m = _re.search(r'(\w+)\s+(\d{1,2}),?\s+(\d{4})', group_title)
                if _m:
                    _mon = _MONTHS.get(_m.group(1).lower())
                    if _mon:
                        try:
                            _parsed_group_date = datetime(int(_m.group(3)), _mon, int(_m.group(2)), tzinfo=timezone.utc)
                        except ValueError:
                            pass
                # Try ISO "YYYY-MM-DD" pattern
                if _parsed_group_date is None:
                    _m2 = _re.search(r'(\d{4})-(\d{2})-(\d{2})', group_title)
                    if _m2:
                        try:
                            _parsed_group_date = datetime(int(_m2.group(1)), int(_m2.group(2)), int(_m2.group(3)), tzinfo=timezone.utc)
                        except ValueError:
                            pass
                if _parsed_group_date is not None and _parsed_group_date > end_date:
                    end_date = _parsed_group_date

            days_to_res = (end_date - now).total_seconds() / 86400.0
            if days_to_res < min_days or days_to_res > max_days:
                continue

            # Fix #1: classify market category
            category = self._classify_market_from_model(market, event_for_market)
            is_sports = category in (CATEGORY_SPORTS, CATEGORY_ESPORTS)
            blocked_keyword = self._first_blocked_keyword(market_text, excluded_keywords) if excluded_keywords else None
            sports_type = getattr(market, "sports_market_type", None)
            is_spread_market = bool(
                block_spread_markets
                and (
                    sports_type == "spreads"
                    or self._is_spread_market(market_text)
                )
            )
            liquidity = safe_float(getattr(market, "liquidity", 0.0))
            liquidity_ok = liquidity >= min_liquidity
            live_game_detected = bool(
                skip_live_games
                and is_sports
                and self._is_live_game(market, now, live_game_buffer_minutes)
            )
            sports_window_ok = (not is_sports) or days_to_res <= sports_max_days

            for outcome in ("YES", "NO"):
                price, bid, ask, token_id = self._extract_side_book(market, prices, outcome)
                if price < min_probability:
                    continue
                if price <= 0.0:
                    continue
                raw_detected_count += 1
                max_settlement_upside_pct = ((1.0 - price) / price) * 100.0

                history_key = f"{market.id}:{outcome}"
                history = history_by_key.get(history_key)
                if not isinstance(history, deque):
                    history = deque(maxlen=max(12, panic_window_points * 2))
                    history_by_key[history_key] = history
                history.append(float(price))
                price_window = list(history)[-panic_window_points:]
                panic_guard_ok = True
                if len(price_window) >= panic_window_points:
                    window_high = max(price_window)
                    window_low = min(price_window)
                    drop_from_high = (window_high - price) / max(window_high, 1e-6)
                    recovery_ratio = (price - window_low) / max(window_high - window_low, 1e-6)
                    if drop_from_high >= panic_drop_threshold and recovery_ratio <= panic_recovery_ratio_max:
                        panic_guard_ok = False

                spread = 0.0
                spread_ok = True
                if isinstance(bid, float) and isinstance(ask, float) and bid > 0.0 and ask > 0.0:
                    spread = max(0.0, ask - bid)
                    if spread > max_spread:
                        spread_ok = False

                target_move = max(min_repricing_buffer, (1.0 - price) * repricing_weight)
                target_price = min(0.999, price + target_move)
                upside_ok = max_settlement_upside_pct >= min_upside_percent

                contract_checks = self._tail_end_checks(
                    source="scanner",
                    strategy_type="tail_end_carry",
                    signal_text=market_text,
                    entry_price=price,
                    dtr=days_to_res,
                    category=category,
                    blocked_keyword=blocked_keyword,
                    is_spread=is_spread_market,
                    liquidity=liquidity,
                    observed_spread=spread,
                    panic_guard_ok=panic_guard_ok,
                    is_live=live_game_detected,
                    limits=limits,
                )
                if not all(check.passed for check in contract_checks):
                    continue

                positions = [
                    {
                        "action": "BUY",
                        "outcome": outcome,
                        "price": price,
                        "max_execution_price": target_price if allow_taker_limit_buy_above_signal else price,
                        "max_entry_price": target_price if allow_taker_limit_buy_above_signal else price,
                        "token_id": token_id,
                        "entry_style": "tail_carry",
                        "price_policy": limits["price_policy"],
                        "time_in_force": limits["time_in_force"],
                        "allow_taker_limit_buy_above_signal": allow_taker_limit_buy_above_signal,
                        "aggressive_limit_buy_submit_as_gtc": bool(limits["aggressive_limit_buy_submit_as_gtc"]),
                        "_tail_end": {
                            "days_to_resolution": days_to_res,
                            "spread": spread,
                            "target_price": target_price,
                            "probability": price,
                            "max_settlement_upside_pct": max_settlement_upside_pct,
                            "market_category": category,
                            "game_start_time": getattr(market, "game_start_time", None),
                            "blocked_keyword": blocked_keyword,
                            "is_spread_market": is_spread_market,
                            "liquidity_ok": liquidity_ok,
                            "sports_window_ok": sports_window_ok,
                            "live_game_detected": live_game_detected,
                            "panic_guard_ok": panic_guard_ok,
                            "spread_ok": spread_ok,
                            "upside_ok": upside_ok,
                        },
                    }
                ]

                market_label = (
                    getattr(market, "group_item_title", None)
                    or getattr(market, "question", None)
                    or ""
                )
                if len(market_label) > 80:
                    market_label = market_label[:77] + "..."

                opp = self.create_opportunity(
                    title=f"Tail Carry: {outcome} {price:.1%} - {market_label}" if market_label else f"Tail Carry: {outcome} {price:.1%} into resolution",
                    description=(
                        f"{outcome} at {price:.3f} with {days_to_res:.1f} days to resolution; "
                        f"target repricing to {target_price:.3f}. [{category}]"
                    ),
                    total_cost=price,
                    expected_payout=target_price,
                    markets=[market],
                    positions=positions,
                    event=event_for_market,
                    is_guaranteed=False,
                    min_liquidity_hard=min_liquidity,
                    min_position_size=max(settings.MIN_POSITION_SIZE, 5.0),
                )
                if not opp:
                    continue

                time_score = 1.0 - (days_to_res / max_days)
                black_swan_penalty = clamp((price - 0.90) * 0.40, 0.0, 0.12)
                spread_penalty = min(0.14, spread * 2.8)
                # Fix #4: higher risk score for sports (reduces sizing via risk penalty in scoring)
                category_penalty = 0.06 if category == CATEGORY_SPORTS else 0.0
                risk = 0.56 - (time_score * 0.10) + black_swan_penalty + spread_penalty + category_penalty
                opp.risk_score = clamp(risk, 0.35, 0.82)
                opp.risk_factors = [
                    f"Near-expiry carry ({days_to_res:.1f} days)",
                    f"Selected probability {price:.1%}",
                    f"Category: {category}",
                    "Non-zero tail-risk remains until resolution",
                ]
                opp.mispricing_type = MispricingType.WITHIN_MARKET

                # Stash category in strategy_context so it flows to exit config
                if not opp.strategy_context:
                    opp.strategy_context = {}
                opp.strategy_context["market_category"] = category
                opp.strategy_context["game_start_time"] = getattr(market, "game_start_time", None)
                opp.strategy_context["sports_market_type"] = getattr(market, "sports_market_type", None)
                opp.strategy_context["blocked_keyword"] = blocked_keyword
                opp.strategy_context["is_spread_market"] = is_spread_market
                opp.strategy_context["execution_min_liquidity"] = min_liquidity
                opp.strategy_context["liquidity_ok"] = liquidity_ok
                opp.strategy_context["sports_min_probability"] = sports_min_probability
                opp.strategy_context["sports_window_ok"] = sports_window_ok
                opp.strategy_context["live_game_detected"] = live_game_detected
                opp.strategy_context["panic_guard_ok"] = panic_guard_ok
                opp.strategy_context["book_spread"] = spread
                opp.strategy_context["book_spread_ok"] = spread_ok
                opp.strategy_context["execution_max_spread"] = max_spread
                opp.strategy_context["max_entry_price"] = target_price if allow_taker_limit_buy_above_signal else price
                opp.strategy_context["execution_max_price"] = target_price if allow_taker_limit_buy_above_signal else price
                opp.strategy_context["allow_taker_limit_buy_above_signal"] = allow_taker_limit_buy_above_signal
                opp.strategy_context["aggressive_limit_buy_submit_as_gtc"] = bool(limits["aggressive_limit_buy_submit_as_gtc"])
                opp.strategy_context["max_settlement_upside_pct"] = max_settlement_upside_pct
                opp.strategy_context["upside_ok"] = upside_ok
                opp.strategy_context["raw_tail_candidate"] = True
                opp.strategy_context["eligibility_checks"] = [asdict(check) for check in contract_checks]

                strength = (target_price - price) * (1.0 - opp.risk_score)
                candidates.append((strength, opp))

        if not candidates:
            self._filter_diagnostics = {
                "strategy_type": self.strategy_type,
                "message": f"Tail carry detected 0/{raw_detected_count} displayable candidates",
                "raw_detected_count": int(raw_detected_count),
                "displayable_count": 0,
                "execution_eligible_count": 0,
                "summary": {
                    "raw_detected_count": int(raw_detected_count),
                    "displayable_count": 0,
                    "execution_eligible_count": 0,
                },
            }
            return []

        candidates.sort(key=lambda item: item[0], reverse=True)
        out: list[Opportunity] = []
        seen: set[tuple[str, str]] = set()
        for _, opp in candidates:
            market_id = str((opp.markets or [{}])[0].get("id") or "")
            outcome = str((opp.positions_to_take or [{}])[0].get("outcome") or "")
            key = (market_id, outcome)
            if key in seen:
                continue
            seen.add(key)
            out.append(opp)
        final_displayable_count = len(out)
        self._filter_diagnostics = {
            "strategy_type": self.strategy_type,
            "message": f"Tail carry detected {final_displayable_count}/{raw_detected_count} displayable candidates",
            "raw_detected_count": int(raw_detected_count),
            "displayable_count": int(final_displayable_count),
            "execution_eligible_count": int(final_displayable_count),
            "summary": {
                "raw_detected_count": int(raw_detected_count),
                "displayable_count": int(final_displayable_count),
                "execution_eligible_count": int(final_displayable_count),
            },
        }
        return out

    # ------------------------------------------------------------------
    # Composable evaluate pipeline overrides
    # ------------------------------------------------------------------

    def custom_checks(self, signal: Any, context: dict, params: dict, payload: dict) -> list[DecisionCheck]:
        """Tail carry: execution-time source, strategy type, entry band, resolution window, and live game checks."""
        limits = self._tail_end_limits(params)
        signal_text = self._signal_market_text(signal, payload)
        strategy_context = getattr(signal, "strategy_context_json", None)
        strategy_context = strategy_context if isinstance(strategy_context, dict) else {}

        source = str(getattr(signal, "source", "") or "").strip().lower()
        strategy_type = (
            str(payload.get("strategy") or payload.get("strategy_type") or getattr(signal, "strategy_type", "") or "")
            .strip()
            .lower()
        )

        entry_price = to_float(getattr(signal, "entry_price", 0.0), 0.0)
        if entry_price <= 0.0:
            positions = payload.get("positions_to_take") if isinstance(payload.get("positions_to_take"), list) else []
            if positions:
                entry_price = to_float((positions[0] or {}).get("price"), 0.0)

        dtr = days_to_resolution(payload)
        max_settlement_upside_pct = ((1.0 - entry_price) / entry_price * 100.0) if entry_price > 0.0 else 0.0
        category = self._classify_market_from_payload(payload)
        is_sports = category in (CATEGORY_SPORTS, CATEGORY_ESPORTS)
        live_game_buffer = float(limits["live_game_buffer_minutes"])
        now = utcnow().astimezone(timezone.utc)
        is_live = bool(limits["skip_live_games"] and is_sports and self._is_live_game_from_payload(payload, now, live_game_buffer))

        payload["_entry_price"] = entry_price
        payload["_dtr"] = dtr
        payload["_max_days"] = float(limits["sports_max_days_to_resolution"] if is_sports else limits["max_days_to_resolution"])
        payload["_strategy_type"] = strategy_type
        payload["_max_settlement_upside_pct"] = max_settlement_upside_pct
        payload["_market_category"] = category
        payload["_is_live_game"] = is_live

        blocked_keyword = self._first_blocked_keyword(signal_text, list(limits.get("exclude_market_keywords") or []))
        if blocked_keyword is None:
            blocked_keyword = str(strategy_context.get("blocked_keyword") or payload.get("blocked_keyword") or "").strip() or None
        if blocked_keyword is not None:
            payload["blocked_keyword"] = blocked_keyword
        liquidity = max(0.0, to_float(getattr(signal, "liquidity", 0.0), 0.0))
        observed_spread = max(
            0.0,
            to_float(strategy_context.get("book_spread", payload.get("book_spread", 0.0)), 0.0),
        )
        panic_guard_ok = bool(strategy_context.get("panic_guard_ok", payload.get("panic_guard_ok", True)))
        is_spread = bool(strategy_context.get("is_spread_market", payload.get("is_spread_market", False)))
        if not is_spread:
            markets_list = payload.get("markets")
            if isinstance(markets_list, list) and markets_list:
                m0 = markets_list[0] if isinstance(markets_list[0], dict) else {}
                if m0.get("sports_market_type") == "spreads":
                    is_spread = True
        if not is_spread:
            is_spread = self._is_spread_market(signal_text)

        checks = self._tail_end_checks(
            source=source,
            strategy_type=strategy_type,
            signal_text=signal_text,
            entry_price=entry_price,
            dtr=dtr,
            category=category,
            blocked_keyword=blocked_keyword,
            is_spread=is_spread,
            liquidity=liquidity,
            observed_spread=observed_spread,
            panic_guard_ok=panic_guard_ok,
            is_live=is_live,
            limits=limits,
        )
        return checks

    def compute_score(
        self, edge: float, confidence: float, risk_score: float, market_count: int, payload: dict
    ) -> float:
        """Tail carry: edge*2.5 + conf*28 + entry*6 - risk*9 + time_bonus."""
        entry_price = float(payload.get("_entry_price", 0) or 0)
        dtr = payload.get("_dtr")
        max_days = float(payload.get("_max_days", 1.0) or 1.0)

        score = (edge * 2.5) + (confidence * 28.0) + (entry_price * 6.0) - (risk_score * 9.0)
        if dtr is not None:
            score += max(0.0, (max_days - dtr) * 0.4)
        return score

    def evaluate(self, signal: Any, context: dict) -> StrategyDecision:
        """Tail carry: composable pipeline with adaptive sizing, category-aware sizing, and custom payload."""
        params = context.get("params") or {}
        payload = signal_payload(signal)

        min_edge = to_float(params.get("min_edge_percent", 1.0), 1.0)
        min_conf = to_confidence(params.get("min_confidence", 0.35), 0.35)
        max_risk = to_confidence(params.get("max_risk_score", 0.78), 0.78)
        base_size, max_size = _trader_size_limits(context)
        sizing_policy = str(params.get("sizing_policy", "adaptive") or "adaptive")

        edge = max(0.0, to_float(getattr(signal, "edge_percent", 0.0), 0.0))
        confidence = to_confidence(getattr(signal, "confidence", 0.0), 0.0)
        liquidity = max(0.0, to_float(getattr(signal, "liquidity", 0.0), 0.0))
        risk_score = to_confidence(payload.get("risk_score", 0.5), 0.5)
        market_count = len(payload.get("markets") or [])

        # Standard checks
        checks = [
            DecisionCheck("edge", "Edge threshold", edge >= min_edge, score=edge, detail=f"min={min_edge:.2f}"),
            DecisionCheck(
                "confidence",
                "Confidence threshold",
                confidence >= min_conf,
                score=confidence,
                detail=f"min={min_conf:.2f}",
            ),
            DecisionCheck(
                "risk", "Risk ceiling", risk_score <= max_risk, score=risk_score, detail=f"max={max_risk:.2f}"
            ),
        ]

        # Strategy-specific checks (also stashes entry_price/dtr/category/etc in payload)
        checks.extend(self.custom_checks(signal, context, params, payload))

        score = self.compute_score(edge, confidence, risk_score, market_count, payload)

        entry_price = float(payload.get("_entry_price", 0) or 0)
        dtr = payload.get("_dtr")
        max_settlement_upside_pct = float(payload.get("_max_settlement_upside_pct", 0.0) or 0.0)
        strategy_type = str(payload.get("_strategy_type", "") or "")
        category = str(payload.get("_market_category", CATEGORY_OTHER) or CATEGORY_OTHER)

        if not all(c.passed for c in checks):
            return StrategyDecision(
                decision="skipped",
                reason="Tail carry filters not met",
                score=score,
                checks=checks,
                payload={
                    "edge": edge,
                    "confidence": confidence,
                    "risk_score": risk_score,
                    "entry_price": entry_price,
                    "days_to_resolution": dtr,
                    "max_settlement_upside_pct": max_settlement_upside_pct,
                    "market_category": category,
                },
            )

        direction = str(getattr(signal, "direction", "") or "")
        probability = selected_probability(signal, payload, direction)

        # Fix #4: reduce position sizes for sports markets
        sports_sizing_multiplier = clamp(
            to_float(params.get("sports_sizing_multiplier", 0.55), 0.55), 0.1, 1.0
        )
        effective_base_size = base_size
        effective_max_size = max_size
        if category == CATEGORY_SPORTS:
            effective_base_size = base_size * sports_sizing_multiplier
            effective_max_size = max_size * sports_sizing_multiplier

        from services.trader_orchestrator.strategies.sizing import compute_position_size

        sizing = compute_position_size(
            base_size_usd=effective_base_size,
            max_size_usd=effective_max_size,
            edge_percent=edge,
            confidence=confidence,
            sizing_policy=sizing_policy,
            probability=probability,
            entry_price=entry_price if entry_price > 0 else None,
            liquidity_usd=liquidity,
            liquidity_cap_fraction=0.06,
        )

        # Stash category metadata into the decision payload so it flows to
        # strategy_exit_config via the session engine's param propagation.
        # We also stash market question/slug for classification in should_exit.
        market_question = str(getattr(signal, "market_question", "") or "")
        market_slug = ""
        event_title = str(payload.get("event_title", "") or "")
        signal_category = str(payload.get("category", "") or "")
        markets_list = payload.get("markets")
        if isinstance(markets_list, list) and markets_list:
            m0 = markets_list[0] if isinstance(markets_list[0], dict) else {}
            market_slug = str(m0.get("slug", "") or "")
            if not market_question:
                market_question = str(m0.get("question", "") or "")

        return StrategyDecision(
            decision="selected",
            reason="Tail carry signal selected",
            score=score,
            size_usd=float(sizing["size_usd"]),
            checks=checks,
            payload={
                "edge": edge,
                "confidence": confidence,
                "risk_score": risk_score,
                "entry_price": entry_price,
                "days_to_resolution": dtr,
                "max_settlement_upside_pct": max_settlement_upside_pct,
                "sizing": sizing,
                "strategy_type": strategy_type,
                "market_category": category,
                # Stash for should_exit classification (flows into strategy_exit_config)
                "_market_category": category,
                "_market_question": market_question,
                "_market_slug": market_slug,
                "_event_title": event_title,
                "_category": signal_category,
                "_sports_market_type": (
                    markets_list[0].get("sports_market_type")
                    if isinstance(markets_list, list) and markets_list and isinstance(markets_list[0], dict)
                    else None
                ),
            },
        )

    # ------------------------------------------------------------------
    # Time-weighted exit helpers (Round 13: tail-leak fixes #1/#2/#3)
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_tw_schedule(raw: Any, fallback: list[list[float]]) -> list[tuple[float, float, float]]:
        """Coerce a TW schedule into ``[(minutes_le, trailing_pct, inversion_floor), ...]``
        sorted ascending by minutes_le. Bad rows are dropped, not rescued.
        """
        rows: list[tuple[float, float, float]] = []
        candidates = raw if isinstance(raw, list) else fallback
        for row in candidates:
            if not isinstance(row, (list, tuple)) or len(row) < 3:
                continue
            try:
                mins = float(row[0])
                trail = float(row[1])
                inv = float(row[2])
            except (TypeError, ValueError):
                continue
            if mins <= 0 or trail <= 0:
                continue
            rows.append((mins, max(0.5, trail), clamp(inv, 0.05, 0.95)))
        if not rows:
            for row in fallback:
                rows.append((float(row[0]), float(row[1]), float(row[2])))
        rows.sort(key=lambda r: r[0])
        return rows

    @classmethod
    def _time_weighted_stop_params(
        cls,
        seconds_left: float | None,
        is_sports: bool,
        config: dict[str, Any],
    ) -> tuple[float | None, float | None]:
        """Return ``(trailing_stop_pct, inversion_floor)`` for the current TTR.

        Returns ``(None, None)`` when ``seconds_left`` is unknown so callers
        fall back to legacy static thresholds.
        """
        if seconds_left is None or seconds_left < 0:
            return None, None
        schedule_key = "sports_time_weighted_stop_schedule" if is_sports else "time_weighted_stop_schedule"
        fallback = (
            cls.default_config.get(schedule_key)
            or cls.default_config["time_weighted_stop_schedule"]
        )
        rows = cls._normalize_tw_schedule(config.get(schedule_key), fallback)
        minutes_left = seconds_left / 60.0
        for mins_le, trail_pct, inv_floor in rows:
            if minutes_left <= mins_le:
                return trail_pct, inv_floor
        # Past the largest bucket: use the loosest one.
        last = rows[-1]
        return last[1], last[2]

    def _velocity_history(self, token_id: str | None) -> deque:
        """Per-token rolling (timestamp, price) deque for velocity detection."""
        bucket = self.state.setdefault("tail_carry_position_velocity", {})
        key = str(token_id or "_unknown")
        history = bucket.get(key)
        if not isinstance(history, deque):
            # maxlen sized to cover the velocity window (default 300s) at the
            # throttled ~1.5s sample spacing (see _velocity_check) with margin,
            # so the lookback anchor is bounded by the time cutoff, not by the
            # deque size — robust to high-frequency exit evaluation (the
            # exit-risk loop evaluates every ~2s + on WS ticks, far more often
            # than reconcile, which would otherwise evict the window's anchor).
            history = deque(maxlen=256)
            bucket[key] = history
        return history

    def _velocity_check(
        self,
        token_id: str | None,
        current_price: float,
        seconds_left: float | None,
        config: dict[str, Any],
    ) -> ExitDecision | None:
        """Return a close decision when price drops too fast in the final window.

        We sample (now, price) into a per-token deque each call. Once we are
        within ``velocity_active_minutes_left`` of resolution, compare the
        oldest sample inside ``velocity_window_seconds`` against the latest.
        A drop of ``velocity_drop_threshold_pct`` or more triggers an exit
        regardless of trailing-stop math — cliff-drops are too fast for it.
        """
        if not _is_bool_true(config.get("velocity_stop_enabled", True)):
            return None
        if current_price <= 0.0:
            return None

        now_ts = utcnow().timestamp()
        history = self._velocity_history(token_id)
        # Throttle sampling to a fixed cadence so the velocity window is
        # frequency-independent: reconcile (~5-15s), the exit-risk loop
        # (~2s), and WS-tick-driven sweeps all collapse to one sample per
        # ~1.5s, keeping the drop-over-window comparison identical regardless
        # of how often should_exit() is invoked.
        if not history or (now_ts - history[-1][0]) >= 1.5:
            history.append((now_ts, float(current_price)))

        active_minutes = max(0.0, safe_float(config.get("velocity_active_minutes_left"), 60.0))
        if seconds_left is None or (seconds_left / 60.0) > active_minutes:
            return None

        window_seconds = max(30.0, safe_float(config.get("velocity_window_seconds"), 300.0))
        threshold_pct = clamp(safe_float(config.get("velocity_drop_threshold_pct"), 3.0), 0.5, 50.0)
        min_samples = max(2, int(safe_float(config.get("velocity_min_samples"), 3)))

        cutoff = now_ts - window_seconds
        in_window = [(ts, px) for ts, px in history if ts >= cutoff]
        if len(in_window) < min_samples:
            return None

        anchor_ts, anchor_price = in_window[0]
        if anchor_price <= 0.0:
            return None
        drop_pct = ((anchor_price - current_price) / anchor_price) * 100.0
        if drop_pct < threshold_pct:
            return None

        return ExitDecision(
            "close",
            (
                f"Velocity stop: {drop_pct:.2f}% drop over "
                f"{int(now_ts - anchor_ts)}s (threshold={threshold_pct:.1f}%, "
                f"minutes_left={(seconds_left or 0)/60.0:.1f})"
            ),
            close_price=current_price,
        )

    def _evaluate_scale_out(
        self,
        position: Any,
        current_price: float,
        seconds_left: float | None,
        config: dict[str, Any],
        strategy_context: dict | None,
    ) -> ExitDecision | None:
        """Resolution-window scale-out ladder (#2).

        Tiers fire once each as the position approaches resolution AND price
        is at or above the tier's ``min_price``. Each fire:
          * emits ``ExitDecision(action="reduce", reduce_fraction=...)`` so
            the backtester does the real partial close, and
          * ratchets ``_scale_out_trailing_floor`` in strategy_context so
            live lifecycle (which currently no-ops "reduce") locks in profit
            via a tight synthetic stop on the next tick.

        The final tier (price-agnostic, T-5m) returns ``close`` for the
        residual to ensure we never sit through the cliff.
        """
        if not _is_bool_true(config.get("scale_out_enabled", True)):
            return None
        if seconds_left is None or seconds_left < 0:
            return None

        tiers_raw = config.get("scale_out_tiers")
        if not isinstance(tiers_raw, list) or not tiers_raw:
            tiers_raw = self.default_config["scale_out_tiers"]

        ctx = strategy_context if isinstance(strategy_context, dict) else {}
        hit: list[int] = list(ctx.get("_scale_out_targets_hit") or [])
        minutes_left = seconds_left / 60.0

        for idx, tier in enumerate(tiers_raw):
            if idx in hit:
                continue
            if not isinstance(tier, dict):
                continue
            try:
                mins_max = float(tier.get("minutes_left_max", 0))
                min_price = float(tier.get("min_price", 0.0))
                fraction = clamp(float(tier.get("exit_fraction", 0.0)), 0.0, 1.0)
            except (TypeError, ValueError):
                continue

            if minutes_left > mins_max:
                continue
            if current_price < min_price:
                continue

            hit.append(idx)
            if isinstance(strategy_context, dict):
                strategy_context["_scale_out_targets_hit"] = hit
            else:
                # Best-effort: stash on the position if context is mutable.
                try:
                    setattr(position, "strategy_context", {"_scale_out_targets_hit": hit})
                except Exception:
                    pass

            tighten_pct_raw = tier.get("tighten_trailing_floor_pct")
            tighten_pct = safe_float(tighten_pct_raw, 0.0) if tighten_pct_raw is not None else 0.0
            if tighten_pct > 0.0 and isinstance(strategy_context, dict):
                new_floor = current_price * (1.0 - (tighten_pct / 100.0))
                prev_floor = safe_float(strategy_context.get("_scale_out_trailing_floor"), 0.0) or 0.0
                if new_floor > prev_floor:
                    strategy_context["_scale_out_trailing_floor"] = new_floor

            reason = (
                f"Scale-out tier {idx + 1}: minutes_left={minutes_left:.1f}<="
                f"{mins_max:.0f}, price={current_price:.4f}>={min_price:.3f}"
            )

            if fraction >= 0.999:
                return ExitDecision("close", reason, close_price=current_price)
            return ExitDecision(
                "reduce",
                reason,
                close_price=current_price,
                reduce_fraction=fraction,
                payload={"scale_out_tier": idx, "tighten_pct": tighten_pct},
            )
        return None

    # ------------------------------------------------------------------
    # Exit logic (Fix #1, #2, #7)
    # ------------------------------------------------------------------

    def should_exit(self, position: Any, market_state: dict) -> ExitDecision:
        """Category-aware exit: trailing stop, resolution proximity hold, inversion stop."""
        if market_state.get("is_resolved"):
            return self.default_exit_check(position, market_state)

        config = getattr(position, "config", None) or {}
        config = dict(config)

        entry_price = safe_float(getattr(position, "entry_price", 0.0), 0.0)
        current_price = safe_float(market_state.get("current_price"), None)
        highest_price = safe_float(getattr(position, "highest_price", None), None)
        age_minutes = safe_float(getattr(position, "age_minutes", None), None)
        seconds_left = safe_float(market_state.get("seconds_left"), None)

        # No price available — hold without blocking default exits so the
        # lifecycle layer can still act on resolution or staleness signals.
        if current_price is None or current_price <= 0.0:
            return ExitDecision("hold", "No current price available for exit evaluation")

        strategy_context = getattr(position, "strategy_context", None)
        if strategy_context is None and hasattr(position, "strategy_context"):
            try:
                position.strategy_context = {}
                strategy_context = position.strategy_context
            except Exception:
                strategy_context = {}

        # Fix #1: determine market category
        category = self._classify_from_exit_config(config, strategy_context)

        is_sports = category in (CATEGORY_SPORTS, CATEGORY_ESPORTS)

        token_id = market_state.get("token_id") if isinstance(market_state, dict) else None
        time_weighted_enabled = _is_bool_true(config.get("time_weighted_stop_enabled", True))
        tw_trailing_pct, tw_inversion_floor = (
            self._time_weighted_stop_params(seconds_left, is_sports, config)
            if time_weighted_enabled
            else (None, None)
        )

        if is_sports:
            hold_inversion_stop_enabled = _is_bool_true(config.get("sports_inversion_stop_enabled", False))
        else:
            hold_inversion_stop_enabled = _is_bool_true(config.get("inversion_stop_enabled", True))
        hold_inversion_floor = clamp(safe_float(config.get("inversion_price_threshold"), 0.50), 0.05, 0.95)
        if is_sports:
            resolution_hold_minutes = max(0.0, safe_float(config.get("sports_resolution_hold_minutes"), 150.0))
        else:
            resolution_hold_minutes = max(0.0, safe_float(config.get("resolution_hold_minutes"), 360.0))
        resolution_hold_max_loss_pct = clamp(
            safe_float(config.get("resolution_hold_max_loss_pct"), 25.0), 5.0, 80.0
        )
        minutes_left = seconds_left / 60.0 if seconds_left is not None else None

        # ── Pre-resolution structural exit (gap cap) ──────────────────────
        # Carrying a favorite INTO resolution exposes the full notional to a
        # binary gap (price holds ~0.9 then snaps to 0 the instant the event
        # resolves — no declining price for a stop to catch).  When enabled,
        # close while still tradable once we are within the configured window
        # of resolution, capturing ~all of the carry without ever holding
        # through the resolution instant.  Checked BEFORE the resolution hold
        # so it overrides the hold.  0 = disabled.
        if is_sports:
            pre_res_minutes = safe_float(config.get("sports_pre_resolution_force_exit_minutes"), 0.0) or 0.0
        else:
            pre_res_minutes = safe_float(config.get("pre_resolution_force_exit_minutes"), 0.0) or 0.0
        if pre_res_minutes > 0.0 and minutes_left is not None and minutes_left <= pre_res_minutes:
            return ExitDecision(
                "close",
                (
                    f"Pre-resolution structural exit ({minutes_left:.1f}m left <= "
                    f"{pre_res_minutes:.0f}m gap cap; price={current_price:.4f}, "
                    f"entry={entry_price:.4f}, category={category})"
                ),
                close_price=current_price,
            )

        in_resolution_hold = (
            _is_bool_true(config.get("resolution_hold_enabled", True))
            and minutes_left is not None
            and minutes_left <= resolution_hold_minutes
        )
        if in_resolution_hold:
            if entry_price > 0.0:
                unrealized_loss_pct = ((entry_price - current_price) / entry_price) * 100.0
                if unrealized_loss_pct >= resolution_hold_max_loss_pct:
                    return ExitDecision(
                        "close",
                        (
                            f"Resolution hold max-loss override ({unrealized_loss_pct:.1f}% loss >= "
                            f"{resolution_hold_max_loss_pct:.0f}% threshold; entry={entry_price:.4f}, "
                            f"current={current_price:.4f}, category={category})"
                        ),
                        close_price=current_price,
                    )
            if hold_inversion_stop_enabled and current_price <= hold_inversion_floor:
                return ExitDecision(
                    "close",
                    (
                        f"Resolution hold confirmed inversion: {current_price:.4f} <= "
                        f"{hold_inversion_floor:.4f} (entry={entry_price:.4f}, category={category}, "
                        f"minutes_left={minutes_left:.1f})"
                    ),
                    close_price=current_price,
                )

            if entry_price <= 0.0 or current_price >= entry_price:
                scale_out_decision = self._evaluate_scale_out(
                    position, current_price, seconds_left, config, strategy_context
                )
                if scale_out_decision is not None:
                    return scale_out_decision

            return ExitDecision(
                "hold",
                (
                    f"Resolution proximity hold ({minutes_left:.0f}m left <= {resolution_hold_minutes:.0f}m; "
                    f"category={category}, price={current_price:.4f}, entry={entry_price:.4f})"
                ),
                payload={"skip_default_exit": True},
            )

        # -- #3: Velocity stop (cliff-drop catcher) --
        # Runs only outside the resolution-hold window; final-window thesis
        # breaks are handled by the hard max-loss and inversion checks above.
        velocity_decision = self._velocity_check(token_id, current_price, seconds_left, config)
        if velocity_decision is not None:
            return velocity_decision

        # -- #2: Scale-out tier evaluation --
        # Outside the final hold, scale-out may still ratchet profit floors.
        scale_out_decision = self._evaluate_scale_out(
            position, current_price, seconds_left, config, strategy_context
        )
        if scale_out_decision is not None and scale_out_decision.action == "close":
            return scale_out_decision
        # If a tier emitted "reduce", we keep the decision and continue: a
        # tier hit may have ratcheted the synthetic floor; we want the
        # subsequent floor-breach check to enforce it on this same tick if
        # the price has already slipped past it.

        # -- #2 (cont): Scale-out floor breach --
        # Once a tier has ratcheted ``_scale_out_trailing_floor``, treat any
        # breach as an immediate close outside the resolution hold.
        scale_out_floor = None
        if isinstance(strategy_context, dict):
            scale_out_floor = safe_float(strategy_context.get("_scale_out_trailing_floor"), None)
            if scale_out_floor is not None and scale_out_floor <= 0.0:
                scale_out_floor = None
        if scale_out_floor is not None and current_price <= scale_out_floor:
            return ExitDecision(
                "close",
                (
                    f"Scale-out floor breach: {current_price:.4f} <= {scale_out_floor:.4f} "
                    f"(category={category}, minutes_left={(seconds_left or 0)/60.0:.1f})"
                ),
                close_price=current_price,
            )

        # If a scale-out tier emitted "reduce" and the floor is not breached
        # this tick, surface it now.
        if scale_out_decision is not None and scale_out_decision.action == "reduce":
            return scale_out_decision

        # -- #1: Inversion stop with time-weighted floor --
        # Static floor (legacy) is the lower bound; the TW schedule can
        # raise it as resolution approaches. We never lower the legacy floor.
        if is_sports:
            inversion_stop_enabled = _is_bool_true(config.get("sports_inversion_stop_enabled", False))
        else:
            inversion_stop_enabled = _is_bool_true(config.get("inversion_stop_enabled", True))
        legacy_inv_floor = clamp(safe_float(config.get("inversion_price_threshold"), 0.50), 0.05, 0.95)
        effective_inv_floor = legacy_inv_floor
        inv_floor_source = "legacy"
        if tw_inversion_floor is not None and tw_inversion_floor > effective_inv_floor:
            effective_inv_floor = tw_inversion_floor
            inv_floor_source = "time_weighted"

        if inversion_stop_enabled and entry_price > 0.0 and current_price > 0.0:
            if current_price <= effective_inv_floor:
                return ExitDecision(
                    "close",
                    (
                        f"Inversion stop ({inv_floor_source}): {current_price:.4f} <= "
                        f"{effective_inv_floor:.4f} (entry={entry_price:.4f}, "
                        f"category={category}, minutes_left="
                        f"{(seconds_left or 0)/60.0:.1f})"
                    ),
                    close_price=current_price,
                )

        # -- #1: Trailing stop with time-weighted pct --
        trailing_stop_enabled = _is_bool_true(config.get("trailing_stop_enabled", True))
        if is_sports:
            legacy_trail_pct = clamp(safe_float(config.get("sports_trailing_stop_pct"), 45.0), 0.5, 80.0)
        else:
            legacy_trail_pct = clamp(safe_float(config.get("trailing_stop_pct"), 25.0), 0.5, 80.0)
        effective_trail_pct = legacy_trail_pct
        trail_source = "legacy"
        if tw_trailing_pct is not None and tw_trailing_pct < effective_trail_pct:
            effective_trail_pct = tw_trailing_pct
            trail_source = "time_weighted"

        if trailing_stop_enabled and highest_price is not None and highest_price > 0.0 and current_price > 0.0:
            drop_from_high_pct = ((highest_price - current_price) / highest_price) * 100.0
            if drop_from_high_pct >= effective_trail_pct:
                return ExitDecision(
                    "close",
                    (
                        f"Trailing stop ({trail_source}): {drop_from_high_pct:.1f}% drop "
                        f"from high {highest_price:.4f}; threshold={effective_trail_pct:.1f}%, "
                        f"current={current_price:.4f}, category={category}, minutes_left="
                        f"{(seconds_left or 0)/60.0:.1f}"
                    ),
                    close_price=current_price,
                )


        # -- Smart take profit (unchanged) --
        smart_enabled_raw = config.get("smart_take_profit_enabled")
        smart_enabled = smart_enabled_raw if isinstance(smart_enabled_raw, bool) else (
            True if smart_enabled_raw is None else _is_bool_true(smart_enabled_raw)
        )
        smart_min_pnl_pct = clamp(safe_float(config.get("smart_take_profit_min_pnl_pct"), 10.0), 0.0, 100.0)
        smart_max_headroom = clamp(safe_float(config.get("smart_take_profit_max_price_headroom"), 0.03), 0.001, 0.20)

        if smart_enabled and entry_price > 0.0 and current_price > 0.0:
            pnl_pct = ((current_price - entry_price) / entry_price) * 100.0
            headroom = max(0.0, 1.0 - current_price)
            if pnl_pct >= smart_min_pnl_pct and headroom <= smart_max_headroom:
                return ExitDecision(
                    "close",
                    (
                        f"Smart take profit near max ({pnl_pct:.1f}% >= {smart_min_pnl_pct:.1f}%, "
                        f"headroom={headroom:.3f})"
                    ),
                    close_price=current_price,
                )

        # -- Max hold timeout --
        max_hold_minutes = safe_float(config.get("max_hold_minutes"), None)
        if (
            max_hold_minutes is not None
            and max_hold_minutes > 0.0
            and age_minutes is not None
            and age_minutes >= max_hold_minutes
            and current_price > 0.0
        ):
            return ExitDecision(
                "close",
                f"Max hold exceeded ({age_minutes:.0f} >= {max_hold_minutes:.0f} min)",
                close_price=current_price,
            )

        stop_loss_config = {
            key: config[key]
            for key in (
                "resolve_only",
                "min_hold_minutes",
                "stop_loss_pct",
                "stop_loss_policy",
                "stop_loss_mode",
                "stop_loss_activation_seconds",
                "stop_loss_near_close_seconds",
            )
            if key in config
        }
        if stop_loss_config:
            stop_loss_position = _StopLossPosition(
                config=stop_loss_config,
                entry_price=entry_price,
                age_minutes=age_minutes,
                highest_price=highest_price,
                outcome_idx=getattr(position, "outcome_idx", 0),
                strategy_context=strategy_context,
            )
            stop_loss_exit = self.default_exit_check(stop_loss_position, market_state)
            if stop_loss_exit.action != "hold":
                return stop_loss_exit

        return ExitDecision(
            "hold",
            f"Tail carry hold — awaiting resolution (category={category})",
            payload={"skip_default_exit": True},
        )

    # ------------------------------------------------------------------
    # Platform gate hooks
    # ------------------------------------------------------------------

    def on_blocked(self, signal, reason: str, context: dict) -> None:
        logger.info("%s: signal blocked — %s (market=%s)", self.name, reason, getattr(signal, "market_id", "?"))

    def on_size_capped(self, original_size: float, capped_size: float, reason: str) -> None:
        logger.info("%s: size capped $%.0f → $%.0f — %s", self.name, original_size, capped_size, reason)

    # ------------------------------------------------------------------
    # Phase 3: declarative pre-submit gates
    # ------------------------------------------------------------------
    #
    # The gates below mirror the cheap (in-memory) subset of the inline
    # checks the strategy already runs in ``custom_checks``/``evaluate``.
    # They run as L0_MEMORY in the orchestrator's pre-submit pipeline so
    # rejections short-circuit BEFORE the L1_CACHED venue gates and
    # BEFORE the DB-write cost.
    #
    # The inline checks are KEPT as a backstop — behaviour is unchanged
    # in production; the gates simply move the same rejection earlier.

    def get_pre_submit_gates(self):
        from services.strategy_sdk import Gate, CostClass, GateContext, GateResult

        params = dict(self.default_config or {})

        def _signal_payload_of(ctx: GateContext) -> dict:
            payload = (ctx.extras or {}).get("signal_payload")
            if isinstance(payload, dict):
                return payload
            payload = getattr(ctx.runtime_signal, "payload_json", None)
            return payload if isinstance(payload, dict) else {}

        def _effective_params(ctx: GateContext) -> dict:
            merged = dict(params)
            merged.update(ctx.strategy_params or {})
            return merged

        def _selected_probability(payload: dict, signal: Any) -> float:
            attr = safe_float(getattr(signal, "selected_probability", None), None)
            if attr is not None and attr > 0.0:
                return float(attr)
            from utils.signal_helpers import selected_probability as _sp
            try:
                return float(_sp(payload) or 0.0)
            except Exception:
                return 0.0

        def excluded_keyword_predicate(ctx: GateContext) -> GateResult:
            payload = _signal_payload_of(ctx)
            cfg = _effective_params(ctx)
            excluded = self._normalize_excluded_keywords(cfg.get("exclude_market_keywords"))
            if not excluded:
                return GateResult(passed=True)
            text_parts = []
            for key in (
                "market_question", "title", "description",
                "market_id", "market_slug", "event_title",
                "event_slug", "category",
            ):
                val = payload.get(key)
                if isinstance(val, str) and val:
                    text_parts.append(val.lower())
            text = " ".join(text_parts)
            blocked = self._first_blocked_keyword(text, excluded)
            if blocked:
                return GateResult(
                    passed=False,
                    reason="tail_carry_excluded_keyword",
                    error_message=f"Market contains excluded keyword: {blocked!r}",
                    detail={"keyword": blocked, "text_len": len(text)},
                )
            return GateResult(passed=True)

        def probability_band_predicate(ctx: GateContext) -> GateResult:
            payload = _signal_payload_of(ctx)
            cfg = _effective_params(ctx)
            min_prob = float(safe_float(cfg.get("min_probability"), 0.85) or 0.0)
            max_prob = float(safe_float(cfg.get("max_probability"), 0.999) or 0.999)
            prob = _selected_probability(payload, ctx.runtime_signal)
            if prob <= 0.0:
                # Missing data ⇒ defer to inline logic.
                return GateResult(passed=True)
            if prob < min_prob or prob > max_prob:
                return GateResult(
                    passed=False,
                    reason="tail_carry_probability_band",
                    error_message=(
                        f"Probability {prob:.3f} outside band "
                        f"[{min_prob:.3f}, {max_prob:.3f}]"
                    ),
                    detail={
                        "probability": prob,
                        "min_probability": min_prob,
                        "max_probability": max_prob,
                    },
                )
            return GateResult(passed=True)

        def min_liquidity_predicate(ctx: GateContext) -> GateResult:
            cfg = _effective_params(ctx)
            min_liq = float(safe_float(cfg.get("min_liquidity"), 0.0) or 0.0)
            if min_liq <= 0.0:
                return GateResult(passed=True)
            liq = safe_float(getattr(ctx.runtime_signal, "liquidity", None), None)
            if liq is None:
                payload = _signal_payload_of(ctx)
                liq = safe_float(payload.get("liquidity"), None)
            if liq is None:
                return GateResult(passed=True)
            if float(liq) < min_liq:
                return GateResult(
                    passed=False,
                    reason="tail_carry_min_liquidity",
                    error_message=f"Liquidity {liq:.0f} < min {min_liq:.0f}",
                    detail={"liquidity": float(liq), "min_liquidity": min_liq},
                )
            return GateResult(passed=True)

        def min_upside_predicate(ctx: GateContext) -> GateResult:
            payload = _signal_payload_of(ctx)
            cfg = _effective_params(ctx)
            min_upside = float(safe_float(cfg.get("min_upside_percent"), 0.0) or 0.0)
            if min_upside <= 0.0:
                return GateResult(passed=True)
            prob = _selected_probability(payload, ctx.runtime_signal)
            if prob <= 0.0:
                return GateResult(passed=True)
            upside_pct = ((1.0 - prob) / prob * 100.0) if prob > 0.0 else 0.0
            if upside_pct < min_upside:
                return GateResult(
                    passed=False,
                    reason="tail_carry_min_upside",
                    error_message=(
                        f"Upside {upside_pct:.2f}% < min {min_upside:.2f}%"
                    ),
                    detail={
                        "upside_percent": round(upside_pct, 3),
                        "min_upside_percent": min_upside,
                        "probability": prob,
                    },
                )
            return GateResult(passed=True)

        return [
            Gate(
                name="tail_carry_excluded_keyword",
                cost_class=CostClass.L0_MEMORY,
                predicate=excluded_keyword_predicate,
            ),
            Gate(
                name="tail_carry_probability_band",
                cost_class=CostClass.L0_MEMORY,
                predicate=probability_band_predicate,
            ),
            Gate(
                name="tail_carry_min_liquidity",
                cost_class=CostClass.L0_MEMORY,
                predicate=min_liquidity_predicate,
            ),
            Gate(
                name="tail_carry_min_upside",
                cost_class=CostClass.L0_MEMORY,
                predicate=min_upside_predicate,
            ),
        ]
