"""
Strategy: Temporal Decay Arbitrage

Markets with time-based questions should follow predictable decay curves,
similar to options theta. When prices deviate from the expected curve,
there's a trading opportunity.

Key insight: A "BTC above $100K by June" market should lose value
as time passes without BTC reaching $100K. If it doesn't decay
as expected, it's mispriced.

Uses a modified Black-Scholes-like decay model adapted for binary outcomes.
"""

import re
import time
from datetime import datetime, timezone
from typing import Optional

from models import Market, Event, ArbitrageOpportunity, StrategyType
from config import settings
from .base import BaseStrategy, utcnow, make_aware
from utils.logger import get_logger

logger = get_logger(__name__)

# Regex patterns to extract deadlines from market questions
_DEADLINE_PATTERNS = [
    # "by December 2025", "by Dec 2025", "by December 31, 2025"
    re.compile(
        r"\bby\s+(\w+\s+\d{1,2},?\s+\d{4}|\w+\s+\d{4})",
        re.IGNORECASE,
    ),
    # "before March 2026", "before Mar 1, 2026"
    re.compile(
        r"\bbefore\s+(\w+\s+\d{1,2},?\s+\d{4}|\w+\s+\d{4})",
        re.IGNORECASE,
    ),
    # "in January 2026", "in Jan 2026"
    re.compile(
        r"\bin\s+(\w+\s+\d{4})",
        re.IGNORECASE,
    ),
    # "by end of 2026", "by the end of 2025"
    re.compile(
        r"\bby\s+(?:the\s+)?end\s+of\s+(\d{4})",
        re.IGNORECASE,
    ),
]

# Month name -> number mapping
_MONTH_MAP = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}

# Regex to detect player stat line patterns like "LaMelo Ball: 4+", "Joel Embiid: 30+"
_PLAYER_STAT_RE = re.compile(r"[A-Z][a-z]+ [A-Z][a-z]+:\s*\d+\+", re.IGNORECASE)

# Default decay rate (square root decay for most markets)
_DEFAULT_DECAY_RATE = 0.5

# Minimum deviation from expected price to flag opportunity (5%)
_MIN_DEVIATION = 0.05

# Only consider markets within 30 days of deadline (steepest decay)
_MAX_DAYS_TO_DEADLINE = 30

# Minimum days to deadline (avoid markets about to expire)
_MIN_DAYS_TO_DEADLINE = 1

# Require at least N observed points before trusting decay deviation.
_MIN_HISTORY_POINTS = 5

# Skip deep-OTM tails where tiny absolute moves produce misleading ROI.
_MIN_ENTRY_PRICE = 0.08
_MAX_ENTRY_PRICE = 0.92

# Minimum expected repricing for a directional trade to be actionable.
_MIN_EXPECTED_MOVE = 0.02

# Certainty shock defaults (all overridable via config/env).
_SHOCK_LOOKBACK_SECONDS = 6 * 3600
_SHOCK_MIN_POINTS = 3
_SHOCK_MAX_DAYS_TO_DEADLINE = 10.0
_SHOCK_MIN_DAYS_TO_DEADLINE = -0.25
_SHOCK_MIN_ABS_MOVE = 0.18
_SHOCK_MAX_RETRACE = 0.12
_SHOCK_MIN_FAVORED_PRICE = 0.55
_SHOCK_MAX_FAVORED_PRICE = 0.97
_SHOCK_TARGET_CERTAINTY = 0.96
_SHOCK_EXTENSION_FACTOR = 0.45
_SHOCK_MIN_EXPECTED_MOVE = 0.03
_SHOCK_MIN_LIQUIDITY_HARD = 1000.0
_SHOCK_MIN_POSITION_SIZE = 50.0

# Known multi-leg sportsbook contracts where temporal decay is not valid.
_MULTILEG_MARKET_PREFIXES = ("KXMVESPORTSMULTIGAMEEXTENDED-",)


class TemporalDecayStrategy(BaseStrategy):
    """
    Temporal Decay Arbitrage: Exploit time-decay mispricing in deadline markets.

    Identifies markets with time-based questions (e.g. "X by [date]") and
    calculates expected price decay using a square-root decay model. When
    actual prices deviate significantly from expected decay, flags as
    opportunity.

    This is a statistical edge strategy, not risk-free arbitrage.
    """

    strategy_type = StrategyType.TEMPORAL_DECAY
    name = "Temporal Decay"
    description = "Exploit time-decay mispricing in deadline markets"

    def __init__(self):
        super().__init__()
        # market_id -> [(timestamp, yes_price)] for tracking decay over time
        self._price_history: dict[str, list[tuple[float, float]]] = {}
        # market_id -> (deadline_dt, first_seen_price) for decay calculation
        self._market_baselines: dict[str, tuple[datetime, float]] = {}

    def detect(
        self, events: list[Event], markets: list[Market], prices: dict[str, dict]
    ) -> list[ArbitrageOpportunity]:
        if not settings.TEMPORAL_DECAY_ENABLED:
            return []

        now = utcnow()
        scan_time = time.time()
        opportunities: list[ArbitrageOpportunity] = []

        for market in markets:
            if market.closed or not market.active:
                continue
            if len(market.outcome_prices) < 2:
                continue

            market_keys = [
                str(getattr(market, "id", "") or "").upper(),
                str(getattr(market, "condition_id", "") or "").upper(),
            ]
            if any(
                key.startswith(prefix)
                for key in market_keys
                for prefix in _MULTILEG_MARKET_PREFIXES
            ):
                continue

            # Skip sports parlays before either decay or shock branches.
            if self._is_sports_parlay(market):
                continue

            yes_price = self._get_live_price(market, prices)
            no_price = self._get_live_no_price(market, prices)

            # Record price history
            if market.id not in self._price_history:
                self._price_history[market.id] = []
            self._price_history[market.id].append((scan_time, yes_price))
            # Keep bounded
            if len(self._price_history[market.id]) > 100:
                self._price_history[market.id] = self._price_history[market.id][-100:]

            # Fast directional branch: detect sudden repricing toward certainty
            # in either direction (YES surge or YES crash -> NO surge).
            shock_opp = self._create_certainty_shock_opportunity(
                market=market,
                yes_price=yes_price,
                no_price=no_price,
                now=now,
                scan_time=scan_time,
            )
            if shock_opp:
                opportunities.append(shock_opp)
                continue

            # Try to extract a deadline from the question
            deadline = self._extract_deadline(market)
            if deadline is None:
                continue

            # Calculate days remaining
            days_remaining = (deadline - now).total_seconds() / 86400.0
            if days_remaining < _MIN_DAYS_TO_DEADLINE:
                continue  # Too close to expiry, too risky
            if days_remaining > _MAX_DAYS_TO_DEADLINE:
                continue  # Too far from deadline, decay is gentle

            # Establish or update baseline for this market.
            # Use the historical maximum observed price as the baseline,
            # which better approximates the initial/peak price than a
            # single first-seen snapshot.
            if market.id not in self._market_baselines:
                initial_price = max(yes_price, 0.10)  # Floor at 0.10
                self._market_baselines[market.id] = (deadline, initial_price)
            else:
                stored_deadline, stored_price = self._market_baselines[market.id]
                # Update baseline if we see a higher price (better peak estimate)
                initial_price = max(stored_price, yes_price)
                if initial_price > stored_price:
                    self._market_baselines[market.id] = (stored_deadline, initial_price)

            if len(self._price_history[market.id]) < _MIN_HISTORY_POINTS:
                continue

            # Calculate total days from first observation to deadline
            first_seen_time = self._price_history[market.id][0][0]
            first_seen_dt = datetime.fromtimestamp(first_seen_time, tz=timezone.utc)
            total_days = max((deadline - first_seen_dt).total_seconds() / 86400.0, 1.0)

            # Expected decay: p_expected = p_initial * (days_remaining / total_days)^decay_rate
            ratio = min(days_remaining / total_days, 1.0)
            p_expected = initial_price * (ratio**_DEFAULT_DECAY_RATE)

            # Compare actual price to expected
            deviation = yes_price - p_expected

            if abs(deviation) < _MIN_DEVIATION:
                continue  # Within normal range

            # Build opportunity
            opp = self._create_decay_opportunity(
                market,
                yes_price,
                p_expected,
                deviation,
                days_remaining,
                deadline,
                prices,
            )
            if opp:
                opportunities.append(opp)

        if opportunities:
            logger.info(
                f"Temporal Decay: found {len(opportunities)} decay mispricing(s)"
            )

        return opportunities

    def _get_live_price(self, market: Market, prices: dict[str, dict]) -> float:
        """Get the best available YES price for a market."""
        yes_price = market.yes_price
        if market.clob_token_ids and len(market.clob_token_ids) > 0:
            token = market.clob_token_ids[0]
            if token in prices:
                yes_price = prices[token].get("mid", yes_price)
        return yes_price

    def _get_live_no_price(self, market: Market, prices: dict[str, dict]) -> float:
        """Get the best available NO price for a market."""
        no_price = market.no_price
        if market.clob_token_ids and len(market.clob_token_ids) > 1:
            token = market.clob_token_ids[1]
            if token in prices:
                no_price = prices[token].get("mid", no_price)
        return no_price

    def _create_certainty_shock_opportunity(
        self,
        market: Market,
        yes_price: float,
        no_price: float,
        now: datetime,
        scan_time: float,
    ) -> Optional[ArbitrageOpportunity]:
        """Create directional opportunity when price rapidly reprices toward certainty."""
        if not getattr(settings, "TEMPORAL_SHOCK_ENABLED", True):
            return None

        history = self._price_history.get(market.id, [])
        min_points = int(getattr(settings, "TEMPORAL_SHOCK_MIN_POINTS", _SHOCK_MIN_POINTS))
        if len(history) < min_points:
            return None

        deadline = self._extract_deadline(market)
        if deadline is None:
            return None

        days_remaining = (deadline - now).total_seconds() / 86400.0
        max_days = float(
            getattr(
                settings,
                "TEMPORAL_SHOCK_MAX_DAYS_TO_DEADLINE",
                _SHOCK_MAX_DAYS_TO_DEADLINE,
            )
        )
        min_days = float(
            getattr(
                settings,
                "TEMPORAL_SHOCK_MIN_DAYS_TO_DEADLINE",
                _SHOCK_MIN_DAYS_TO_DEADLINE,
            )
        )
        if days_remaining > max_days or days_remaining < min_days:
            return None

        lookback_seconds = int(
            getattr(
                settings,
                "TEMPORAL_SHOCK_LOOKBACK_SECONDS",
                _SHOCK_LOOKBACK_SECONDS,
            )
        )
        cutoff = scan_time - lookback_seconds
        window_prices = [p for ts, p in history if ts >= cutoff]
        if len(window_prices) < min_points:
            window_prices = [p for _, p in history[-min_points:]]
        if len(window_prices) < min_points:
            return None

        peak = max(window_prices)
        trough = min(window_prices)
        up_move = yes_price - trough
        down_move = peak - yes_price
        min_abs_move = float(
            getattr(settings, "TEMPORAL_SHOCK_MIN_ABS_MOVE", _SHOCK_MIN_ABS_MOVE)
        )
        if up_move < min_abs_move and down_move < min_abs_move:
            return None

        yes_token = market.clob_token_ids[0] if market.clob_token_ids else None
        no_token = (
            market.clob_token_ids[1]
            if market.clob_token_ids and len(market.clob_token_ids) > 1
            else None
        )

        if up_move >= down_move:
            outcome = "YES"
            entry_price = yes_price
            token_id = yes_token
            move = up_move
            retrace = max(peak - yes_price, 0.0)
            shock_desc = "YES repricing upward"
        else:
            outcome = "NO"
            entry_price = no_price
            token_id = no_token
            move = down_move
            retrace = max(yes_price - trough, 0.0)
            shock_desc = "YES repricing downward (NO upward)"

        max_retrace = float(
            getattr(settings, "TEMPORAL_SHOCK_MAX_RETRACE", _SHOCK_MAX_RETRACE)
        )
        if retrace > max_retrace:
            return None

        min_favored = float(
            getattr(
                settings,
                "TEMPORAL_SHOCK_MIN_FAVORED_PRICE",
                _SHOCK_MIN_FAVORED_PRICE,
            )
        )
        max_favored = float(
            getattr(
                settings,
                "TEMPORAL_SHOCK_MAX_FAVORED_PRICE",
                _SHOCK_MAX_FAVORED_PRICE,
            )
        )
        if entry_price < min_favored or entry_price > max_favored:
            return None

        target_floor = float(
            getattr(
                settings,
                "TEMPORAL_SHOCK_TARGET_CERTAINTY",
                _SHOCK_TARGET_CERTAINTY,
            )
        )
        extension = float(
            getattr(
                settings,
                "TEMPORAL_SHOCK_EXTENSION_FACTOR",
                _SHOCK_EXTENSION_FACTOR,
            )
        )
        target_exit_price = max(target_floor, entry_price + move * extension)
        target_exit_price = max(0.01, min(0.995, target_exit_price))
        expected_move = target_exit_price - entry_price

        min_expected_move = float(
            getattr(
                settings,
                "TEMPORAL_SHOCK_MIN_EXPECTED_MOVE",
                _SHOCK_MIN_EXPECTED_MOVE,
            )
        )
        if expected_move < min_expected_move:
            return None

        question_short = market.question[:50]
        positions = [
            {
                "action": "BUY",
                "outcome": outcome,
                "market": question_short,
                "price": entry_price,
                "token_id": token_id,
                "rationale": (
                    f"{shock_desc}; lookback move {move:.3f}, "
                    f"retrace {retrace:.3f}, target ${target_exit_price:.3f}"
                ),
            },
        ]

        min_liquidity_hard = float(
            getattr(
                settings,
                "TEMPORAL_SHOCK_MIN_LIQUIDITY_HARD",
                _SHOCK_MIN_LIQUIDITY_HARD,
            )
        )
        min_position_size = float(
            getattr(
                settings,
                "TEMPORAL_SHOCK_MIN_POSITION_SIZE",
                _SHOCK_MIN_POSITION_SIZE,
            )
        )

        opp = self.create_opportunity(
            title=f"Certainty Shock: {question_short}...",
            description=(
                f"Rapid repricing detected near deadline ({days_remaining:.2f}d). "
                f"{shock_desc}: peak=${peak:.3f}, trough=${trough:.3f}, "
                f"current YES=${yes_price:.3f}. "
                f"Buy {outcome} @ ${entry_price:.3f}, "
                f"target repricing ${target_exit_price:.3f}."
            ),
            total_cost=entry_price,
            expected_payout=target_exit_price,
            markets=[market],
            positions=positions,
            is_guaranteed=False,
            min_liquidity_hard=min_liquidity_hard,
            min_position_size=min_position_size,
        )

        if opp and opp.roi_percent > 120.0:
            return None

        if opp:
            risk_score = 0.62 - min(move * 0.35, 0.20)
            if days_remaining <= 1.0:
                risk_score -= 0.05
            opp.risk_score = max(0.35, min(risk_score, 0.75))
            opp.risk_factors.insert(
                0,
                "DIRECTIONAL BET — certainty shock can reverse before final settlement.",
            )
            opp.risk_factors.append(
                f"Certainty shock: {shock_desc}, move={move:.1%}, retrace={retrace:.1%}"
            )
            opp.risk_factors.append(
                f"Near expiry window: {days_remaining:.2f} days to deadline"
            )
            opp.risk_factors.append(
                f"Target repricing edge: +${expected_move:.3f} per share"
            )

        return opp

    def _is_sports_parlay(self, market: Market) -> bool:
        """Check if a market is a sports parlay where temporal decay is invalid.

        Sports outcomes are discrete events (team wins or loses), so the
        continuous time-decay model produces garbage results.  We check
        the market question for league names, team names, player stat
        patterns, spread/line syntax, and other sport-specific keywords.
        """
        q = market.question.lower()

        # Generic multi-leg detection catches many sportsbook contracts
        # before sport-specific keyword checks.
        if q.count("yes ") + q.count("no ") >= 2:
            return True
        if q.count(",") >= 2:
            return True

        # Check sport league / general sport keywords
        sport_keywords = [
            "nba",
            "nfl",
            "mlb",
            "nhl",
            "ufc",
            "mma",
            "boxing",
            "ncaa",
            "march madness",
            "college basketball",
            "ncaab",
            "college",
            "premier league",
            "la liga",
            "bundesliga",
            "serie a",
            "champions league",
            "mls",
            "ligue 1",
            "epl",
            "moneyline",
            "parlay",
            "spread",
            "halftime",
            "quarter",
            "inning",
            "period",
            "overtime",
            "points",
            "rebounds",
            "assists",
            "touchdowns",
            "yards",
            "goals scored",
        ]
        if any(kw in q for kw in sport_keywords):
            return True

        # Check for team names (NBA + NFL + popular college + soccer)
        team_keywords = [
            # NBA
            "lakers",
            "celtics",
            "76ers",
            "sixers",
            "warriors",
            "bucks",
            "heat",
            "nuggets",
            "suns",
            "clippers",
            "nets",
            "knicks",
            "bulls",
            "cavaliers",
            "mavericks",
            "timberwolves",
            "pelicans",
            "thunder",
            "grizzlies",
            "hawks",
            "pacers",
            "raptors",
            "magic",
            "wizards",
            "hornets",
            "pistons",
            "kings",
            "blazers",
            "spurs",
            "jazz",
            "rockets",
            # NFL
            "chiefs",
            "49ers",
            "eagles",
            "cowboys",
            "dolphins",
            "bills",
            "ravens",
            "bengals",
            "steelers",
            "lions",
            "packers",
            "bears",
            "vikings",
            "saints",
            "buccaneers",
            "falcons",
            "panthers",
            "seahawks",
            "cardinals",
            "rams",
            "commanders",
            "giants",
            "jets",
            "patriots",
            "broncos",
            "raiders",
            "chargers",
            "titans",
            "colts",
            "jaguars",
            "texans",
            "browns",
            # College
            "bucknell",
            "duke",
            "kentucky",
            "gonzaga",
            "villanova",
            "michigan",
            "ohio state",
            "alabama",
            "clemson",
        ]
        if any(team in q for team in team_keywords):
            return True

        # Check for player stat patterns like "LaMelo Ball: 4+"
        if _PLAYER_STAT_RE.search(market.question):
            return True

        return False

    def _extract_deadline(self, market: Market) -> Optional[datetime]:
        """
        Extract a deadline datetime from the market question text.

        First checks the market's end_date field, then tries regex extraction
        from the question text.
        """
        # Prefer the market's own end_date if available
        if market.end_date:
            return make_aware(market.end_date)

        question = market.question

        for pattern in _DEADLINE_PATTERNS:
            match = pattern.search(question)
            if match:
                date_str = match.group(1).strip().rstrip(",")
                parsed = self._parse_date_string(date_str)
                if parsed:
                    return parsed

        return None

    def _parse_date_string(self, date_str: str) -> Optional[datetime]:
        """Parse a date string extracted from a market question."""
        parts = date_str.lower().split()

        if len(parts) == 1:
            # Just a year like "2026"
            try:
                year = int(parts[0])
                return datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
            except ValueError:
                return None

        if len(parts) >= 2:
            month_str = parts[0]
            month = _MONTH_MAP.get(month_str)
            if month is None:
                return None

            if len(parts) == 2:
                # "March 2026" -> end of that month
                try:
                    year = int(parts[1])
                    import calendar

                    _, day = calendar.monthrange(year, month)
                    return datetime(year, month, day, 23, 59, 59, tzinfo=timezone.utc)
                except ValueError:
                    return None

            if len(parts) >= 3:
                # "March 15 2026" or "March 15, 2026"
                try:
                    day_str = parts[1].rstrip(",")
                    day = int(day_str)
                    year = int(parts[2])
                    return datetime(year, month, day, 23, 59, 59, tzinfo=timezone.utc)
                except ValueError:
                    return None

        return None

    def _create_decay_opportunity(
        self,
        market: Market,
        actual_price: float,
        expected_price: float,
        deviation: float,
        days_remaining: float,
        deadline: datetime,
        prices: dict[str, dict],
    ) -> Optional[ArbitrageOpportunity]:
        """Create an opportunity from a temporal decay deviation."""
        no_price = market.no_price
        if market.clob_token_ids and len(market.clob_token_ids) > 1:
            token = market.clob_token_ids[1]
            if token in prices:
                no_price = prices[token].get("mid", no_price)

        if deviation > 0:
            # Market is OVERPRICED relative to decay expectation -> buy NO
            action = "BUY"
            outcome = "NO"
            entry_price = no_price
            token_id = (
                market.clob_token_ids[1]
                if market.clob_token_ids and len(market.clob_token_ids) > 1
                else None
            )
            direction_desc = "overpriced"
        else:
            # Market is UNDERPRICED relative to decay expectation -> buy YES
            action = "BUY"
            outcome = "YES"
            entry_price = actual_price
            token_id = market.clob_token_ids[0] if market.clob_token_ids else None
            direction_desc = "underpriced"

        total_cost = entry_price

        # Skip extreme prices
        if entry_price < _MIN_ENTRY_PRICE or entry_price > _MAX_ENTRY_PRICE:
            return None

        if deviation > 0:
            target_exit_price = max(0.01, min(0.99, 1.0 - expected_price))
        else:
            target_exit_price = max(0.01, min(0.99, expected_price))

        expected_move = target_exit_price - entry_price
        if expected_move < _MIN_EXPECTED_MOVE:
            return None

        # Risk score: 0.55 - 0.65
        # Closer to deadline = steeper decay = higher confidence = lower risk
        base_risk = 0.65
        # Reduce risk as deviation increases (stronger signal)
        deviation_adjustment = min(abs(deviation) * 1.5, 0.10)
        risk_score = max(base_risk - deviation_adjustment, 0.55)

        question_short = market.question[:50]
        positions = [
            {
                "action": action,
                "outcome": outcome,
                "market": question_short,
                "price": entry_price,
                "token_id": token_id,
                "rationale": (
                    f"Expected decay price: ${expected_price:.3f}, "
                    f"actual: ${actual_price:.3f} "
                    f"({direction_desc} by {abs(deviation):.3f})"
                ),
            },
        ]

        opp = self.create_opportunity(
            title=f"Temporal Decay: {question_short}...",
            description=(
                f"Deadline market {direction_desc} vs expected decay curve. "
                f"YES actual: ${actual_price:.3f}, expected: ${expected_price:.3f} "
                f"(deviation: {abs(deviation):.3f}). "
                f"{days_remaining:.1f} days to deadline. "
                f"Buy {outcome} at ${entry_price:.3f}, "
                f"target re-price ${target_exit_price:.3f}."
            ),
            total_cost=total_cost,
            expected_payout=target_exit_price,
            markets=[market],
            positions=positions,
            is_guaranteed=False,
            min_liquidity_hard=1500.0,
            min_position_size=50.0,
        )

        if opp and opp.roi_percent > 120.0:
            return None

        if opp:
            # Override risk score to our statistical range
            opp.risk_score = risk_score
            opp.risk_factors.append(
                f"Statistical edge (not risk-free): "
                f"decay deviation {abs(deviation):.1%}"
            )
            opp.risk_factors.append(
                f"Deadline in {days_remaining:.0f} days "
                f"({deadline.strftime('%Y-%m-%d')})"
            )
            opp.risk_factors.append(
                f"Expected repricing target: +${expected_move:.3f} per share"
            )
            opp.risk_factors.insert(
                0,
                "DIRECTIONAL BET — not arbitrage. "
                "Decay deviation may reflect new information, not mispricing.",
            )
            if days_remaining < 7:
                opp.risk_factors.append(
                    "Near-deadline: steep decay but higher event uncertainty"
                )

        return opp
