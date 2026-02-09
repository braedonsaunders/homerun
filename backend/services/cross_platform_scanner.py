import asyncio
import re
import string
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Optional
from datetime import datetime, timedelta

from models import Event, Market, ArbitrageOpportunity
from models.opportunity import StrategyType
from services.polymarket import polymarket_client
from services.kalshi_client import kalshi_client
from utils.logger import get_logger

logger = get_logger("cross_platform_scanner")

# Fee schedules (conservative estimates)
POLYMARKET_FEE = 0.02  # 2 % winner-take fee
# Kalshi fee schedule is tiered by volume. Using a conservative mid-tier
# estimate rather than the worst-case 7%.
# Tier breakdown: <$100 = 7%, $100-$999 = 5%, $1K-$9.9K = 3%, $10K+ = 1%
KALSHI_FEE = 0.05  # 5% default (mid-tier conservative estimate)
KALSHI_FEE_TIERS = [
    (10_000, 0.01),  # $10K+ volume: 1%
    (1_000, 0.03),  # $1K-$9.9K volume: 3%
    (100, 0.05),  # $100-$999 volume: 5%
    (0, 0.07),  # <$100 volume: 7%
]


def kalshi_fee_for_volume(volume: float) -> float:
    """Return the Kalshi fee rate for a given volume tier."""
    for threshold, rate in KALSHI_FEE_TIERS:
        if volume >= threshold:
            return rate
    return 0.07


# ====================================================================== #
#  Domain-specific normalisation maps
# ====================================================================== #

# Common abbreviation expansions used across platforms
_ABBREVIATIONS: dict[str, str] = {
    "btc": "bitcoin",
    "eth": "ethereum",
    "sol": "solana",
    "xrp": "ripple",
    "doge": "dogecoin",
    "bnb": "binance coin",
    "ada": "cardano",
    "dot": "polkadot",
    "avax": "avalanche",
    "matic": "polygon",
    "gop": "republican",
    "dem": "democrat",
    "dems": "democrats",
    "rep": "republican",
    "reps": "republicans",
    "potus": "president of the united states",
    "scotus": "supreme court",
    "gdp": "gross domestic product",
    "cpi": "consumer price index",
    "fed": "federal reserve",
    "eod": "end of day",
    "eoy": "end of year",
    "eom": "end of month",
    "q1": "first quarter",
    "q2": "second quarter",
    "q3": "third quarter",
    "q4": "fourth quarter",
    "nfl": "national football league",
    "nba": "national basketball association",
    "mlb": "major league baseball",
    "nhl": "national hockey league",
    "mls": "major league soccer",
    "ufc": "ultimate fighting championship",
    "f1": "formula 1",
}

# Sports team name mappings: canonical name -> set of aliases (all lower-case)
_TEAM_ALIASES: dict[str, frozenset[str]] = {
    "new york yankees": frozenset({"nyy", "yankees", "ny yankees", "yanks"}),
    "new york mets": frozenset({"nym", "mets", "ny mets"}),
    "los angeles dodgers": frozenset({"lad", "dodgers", "la dodgers"}),
    "los angeles lakers": frozenset({"lal", "lakers", "la lakers"}),
    "los angeles clippers": frozenset({"lac", "clippers", "la clippers"}),
    "los angeles rams": frozenset({"lar", "rams", "la rams"}),
    "los angeles chargers": frozenset({"lac", "chargers", "la chargers"}),
    "los angeles angels": frozenset({"laa", "angels", "la angels", "anaheim angels"}),
    "san francisco 49ers": frozenset({"sf", "49ers", "niners", "san francisco"}),
    "san francisco giants": frozenset({"sfg", "sf giants", "giants"}),
    "new york giants": frozenset({"nyg", "ny giants"}),
    "new york jets": frozenset({"nyj", "jets", "ny jets"}),
    "new york knicks": frozenset({"nyk", "knicks", "ny knicks"}),
    "brooklyn nets": frozenset({"bkn", "nets"}),
    "golden state warriors": frozenset({"gsw", "warriors", "dubs"}),
    "boston celtics": frozenset({"bos", "celtics"}),
    "boston red sox": frozenset({"bos", "red sox", "redsox"}),
    "chicago bulls": frozenset({"chi", "bulls"}),
    "chicago cubs": frozenset({"chc", "cubs"}),
    "chicago white sox": frozenset({"chw", "white sox"}),
    "chicago bears": frozenset({"chi", "bears"}),
    "dallas cowboys": frozenset({"dal", "cowboys"}),
    "dallas mavericks": frozenset({"dal", "mavs", "mavericks"}),
    "green bay packers": frozenset({"gb", "packers"}),
    "kansas city chiefs": frozenset({"kc", "chiefs"}),
    "miami heat": frozenset({"mia", "heat"}),
    "miami dolphins": frozenset({"mia", "dolphins"}),
    "philadelphia eagles": frozenset({"phi", "eagles", "philly eagles"}),
    "philadelphia 76ers": frozenset({"phi", "76ers", "sixers", "philly"}),
    "phoenix suns": frozenset({"phx", "suns"}),
    "tampa bay buccaneers": frozenset({"tb", "bucs", "buccaneers"}),
    "denver broncos": frozenset({"den", "broncos"}),
    "denver nuggets": frozenset({"den", "nuggets"}),
    "houston texans": frozenset({"hou", "texans"}),
    "houston rockets": frozenset({"hou", "rockets"}),
    "houston astros": frozenset({"hou", "astros"}),
    "milwaukee bucks": frozenset({"mil", "bucks"}),
    "milwaukee brewers": frozenset({"mil", "brewers"}),
    "minnesota timberwolves": frozenset({"min", "timberwolves", "wolves", "twolves"}),
    "minnesota vikings": frozenset({"min", "vikings"}),
    "seattle seahawks": frozenset({"sea", "seahawks"}),
    "washington commanders": frozenset({"was", "commanders"}),
    "atlanta hawks": frozenset({"atl", "hawks"}),
    "atlanta braves": frozenset({"atl", "braves"}),
    "atlanta falcons": frozenset({"atl", "falcons"}),
    "toronto raptors": frozenset({"tor", "raptors"}),
    "toronto blue jays": frozenset({"tor", "blue jays", "jays"}),
    "detroit lions": frozenset({"det", "lions"}),
    "detroit pistons": frozenset({"det", "pistons"}),
    "detroit tigers": frozenset({"det", "tigers"}),
    "manchester united": frozenset({"man utd", "man united", "mufc", "united"}),
    "manchester city": frozenset({"man city", "mcfc", "city"}),
    "real madrid": frozenset({"real", "madrid", "rmcf"}),
    "barcelona": frozenset({"barca", "fcb", "fc barcelona"}),
    "liverpool": frozenset({"lfc", "liverpool fc"}),
    "arsenal": frozenset({"afc", "arsenal fc", "gunners"}),
    "chelsea": frozenset({"cfc", "chelsea fc", "blues"}),
}

# Reverse lookup: alias -> canonical name (built once at import time)
_ALIAS_TO_CANONICAL: dict[str, str] = {}
for _canonical, _aliases in _TEAM_ALIASES.items():
    for _alias in _aliases:
        _ALIAS_TO_CANONICAL[_alias] = _canonical
    # Also map the canonical name to itself
    _ALIAS_TO_CANONICAL[_canonical] = _canonical

# Crypto asset normalisation
_CRYPTO_ALIASES: dict[str, str] = {
    "bitcoin": "BTC",
    "btc": "BTC",
    "ethereum": "ETH",
    "eth": "ETH",
    "ether": "ETH",
    "solana": "SOL",
    "sol": "SOL",
    "ripple": "XRP",
    "xrp": "XRP",
    "dogecoin": "DOGE",
    "doge": "DOGE",
    "binance coin": "BNB",
    "bnb": "BNB",
    "cardano": "ADA",
    "ada": "ADA",
    "polkadot": "DOT",
    "dot": "DOT",
    "avalanche": "AVAX",
    "avax": "AVAX",
    "polygon": "MATIC",
    "matic": "MATIC",
    "litecoin": "LTC",
    "ltc": "LTC",
    "chainlink": "LINK",
    "link": "LINK",
    "uniswap": "UNI",
    "uni": "UNI",
}


# ====================================================================== #
#  MatchResult dataclass
# ====================================================================== #


@dataclass
class MatchResult:
    """Result of the cross-platform market matching pipeline.

    Captures every dimension of similarity so that downstream consumers
    (the arbitrage calculator, risk engine, or a human reviewer) can
    inspect *why* two markets were considered equivalent.
    """

    poly_market: Market
    kalshi_market: Market
    text_similarity: float = 0.0
    entity_overlap: float = 0.0
    resolution_match: bool = False
    overall_confidence: float = 0.0
    match_reasoning: str = ""

    # Detailed entity information for debugging / display
    shared_entities: list[str] = field(default_factory=list)
    poly_only_entities: list[str] = field(default_factory=list)
    kalshi_only_entities: list[str] = field(default_factory=list)


# ====================================================================== #
#  Text normalisation helpers
# ====================================================================== #

# Pre-compiled patterns used by _normalize_text
_RE_PUNCTUATION = re.compile(f"[{re.escape(string.punctuation)}]")
_RE_MULTI_SPACE = re.compile(r"\s+")


def _normalize_text(text: str) -> str:
    """Normalise a market question for comparison.

    Pipeline:
        1. Lower-case
        2. Expand known abbreviations
        3. Remove punctuation
        4. Collapse whitespace
    """
    text = text.lower().strip()

    # Expand abbreviations (whole-word only)
    for abbr, expansion in _ABBREVIATIONS.items():
        text = re.sub(rf"\b{re.escape(abbr)}\b", expansion, text)

    text = _RE_PUNCTUATION.sub(" ", text)
    text = _RE_MULTI_SPACE.sub(" ", text).strip()
    return text


def _fuzzy_similarity(a: str, b: str) -> float:
    """Return SequenceMatcher ratio on normalised texts (0..1)."""
    norm_a = _normalize_text(a)
    norm_b = _normalize_text(b)
    return SequenceMatcher(None, norm_a, norm_b).ratio()


# ====================================================================== #
#  Entity extraction
# ====================================================================== #

# Patterns for extracting structured entities from market questions
_RE_PRICE_TARGET = re.compile(
    r"\$\s?([\d,]+(?:\.\d+)?)\s*"
    r"([kmbt](?:illion|housand|rillion)?\b)?",
    re.IGNORECASE,
)
_RE_PERCENTAGE = re.compile(r"([\d.]+)\s*%")
_RE_DATE_MDY = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|"
    r"october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|"
    r"oct|nov|dec)\s+(\d{1,2})(?:st|nd|rd|th)?(?:\s*,?\s*(\d{4}))?\b",
    re.IGNORECASE,
)
_RE_DATE_YEAR = re.compile(r"\b(20\d{2})\b")
_RE_DATE_BY = re.compile(
    r"\bby\s+(end\s+of\s+)?(day|week|month|year|q[1-4]|"
    r"january|february|march|april|may|june|july|august|september|"
    r"october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|"
    r"oct|nov|dec)\b",
    re.IGNORECASE,
)
_RE_NUMBER_THRESHOLD = re.compile(
    r"\b(above|below|over|under|more than|less than|at least|exceed|reach|hit)\s+"
    r"\$?([\d,]+(?:\.\d+)?)\s*"
    r"([kmbt](?:illion|housand|rillion)?\b)?",
    re.IGNORECASE,
)

# Simple capitalized-word heuristic for person/org names.  We look for
# sequences of 2+ capitalized words that are NOT common English words.
_COMMON_TITLE_WORDS = frozenset(
    {
        "Will",
        "The",
        "How",
        "What",
        "When",
        "Where",
        "Which",
        "Who",
        "Does",
        "Did",
        "Can",
        "Could",
        "Would",
        "Should",
        "May",
        "Is",
        "Are",
        "Was",
        "Were",
        "Has",
        "Have",
        "Had",
        "Do",
        "Be",
        "By",
        "For",
        "And",
        "But",
        "Yes",
        "No",
        "Or",
        "Not",
        "Before",
        "After",
        "Price",
        "Win",
        "Reach",
        "Above",
        "Below",
        "Over",
        "Under",
        "End",
        "Day",
        "Week",
        "Month",
        "Year",
        "First",
        "Last",
        "Next",
    }
)

_RE_CAPITALIZED_SEQUENCE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")

# Event type keywords
_EVENT_KEYWORDS: dict[str, str] = {
    "election": "election",
    "vote": "election",
    "ballot": "election",
    "primary": "election",
    "nominee": "election",
    "inaugurate": "election",
    "inauguration": "election",
    "game": "sports",
    "match": "sports",
    "championship": "sports",
    "tournament": "sports",
    "playoff": "sports",
    "playoffs": "sports",
    "super bowl": "sports",
    "world series": "sports",
    "finals": "sports",
    "mvp": "sports",
    "price": "price_movement",
    "trading": "price_movement",
    "close": "price_movement",
    "closing": "price_movement",
    "market cap": "price_movement",
    "ath": "price_movement",
    "all-time high": "price_movement",
    "approval rating": "polling",
    "poll": "polling",
    "favorability": "polling",
    "rate cut": "monetary_policy",
    "rate hike": "monetary_policy",
    "interest rate": "monetary_policy",
    "fed funds": "monetary_policy",
    "federal reserve": "monetary_policy",
    "recession": "economic",
    "inflation": "economic",
    "unemployment": "economic",
    "gdp": "economic",
    "gross domestic product": "economic",
}

# Resolution-criteria phrases that may differ between platforms
_RESOLUTION_KEYWORDS: dict[str, str] = {
    "closing price": "closing_price",
    "close price": "closing_price",
    "at close": "closing_price",
    "at market close": "closing_price",
    "at any point": "any_point",
    "at any time": "any_point",
    "intraday": "any_point",
    "during the day": "any_point",
    "at any point during": "any_point",
    "end of day": "end_of_day",
    "by eod": "end_of_day",
    "11:59 pm": "end_of_day",
    "midnight": "end_of_day",
    "end of year": "end_of_year",
    "december 31": "end_of_year",
    "by year end": "end_of_year",
}


@dataclass
class ExtractedEntities:
    """Entities extracted from a market question."""

    persons: set[str] = field(default_factory=set)
    organizations: set[str] = field(default_factory=set)
    teams: set[str] = field(default_factory=set)  # canonical team names
    crypto_assets: set[str] = field(default_factory=set)  # canonical tickers
    dates: set[str] = field(default_factory=set)  # normalised date strings
    timeframes: set[str] = field(default_factory=set)  # "by end of year", etc.
    price_targets: list[float] = field(default_factory=list)
    percentages: list[float] = field(default_factory=list)
    number_thresholds: list[tuple[str, float]] = field(
        default_factory=list
    )  # (direction, value)
    event_types: set[str] = field(default_factory=set)
    resolution_methods: set[str] = field(default_factory=set)

    def all_entity_strings(self) -> set[str]:
        """Return a flat set of all entity string representations for overlap
        computation."""
        items: set[str] = set()
        items.update(self.persons)
        items.update(self.organizations)
        items.update(self.teams)
        items.update(self.crypto_assets)
        items.update(self.dates)
        items.update(self.timeframes)
        items.update(self.event_types)
        for pt in self.price_targets:
            items.add(f"price:{pt}")
        for pct in self.percentages:
            items.add(f"pct:{pct}")
        for direction, val in self.number_thresholds:
            items.add(f"threshold:{direction}:{val}")
        return items


def _extract_entities(text: str) -> ExtractedEntities:
    """Extract structured entities from a market question string."""
    entities = ExtractedEntities()
    lower = text.lower()

    # --- Crypto assets ---
    for token, canonical in _CRYPTO_ALIASES.items():
        if re.search(rf"\b{re.escape(token)}\b", lower):
            entities.crypto_assets.add(canonical)

    # --- Sports teams ---
    for alias, canonical in _ALIAS_TO_CANONICAL.items():
        if re.search(rf"\b{re.escape(alias)}\b", lower):
            entities.teams.add(canonical)

    # --- Price targets ---
    for m in _RE_PRICE_TARGET.finditer(text):
        val_str = m.group(1).replace(",", "")
        val = float(val_str)
        suffix = (m.group(2) or "").lower()
        if suffix.startswith("k") or suffix.startswith("thousand"):
            val *= 1_000
        elif suffix.startswith("m") or suffix.startswith("million"):
            val *= 1_000_000
        elif suffix.startswith("b") or suffix.startswith("billion"):
            val *= 1_000_000_000
        elif suffix.startswith("t") or suffix.startswith("trillion"):
            val *= 1_000_000_000_000
        entities.price_targets.append(val)

    # --- Percentages ---
    for m in _RE_PERCENTAGE.finditer(text):
        entities.percentages.append(float(m.group(1)))

    # --- Number thresholds (above/below X) ---
    for m in _RE_NUMBER_THRESHOLD.finditer(text):
        direction = m.group(1).lower()
        val_str = m.group(2).replace(",", "")
        val = float(val_str)
        suffix = (m.group(3) or "").lower()
        if suffix.startswith("k") or suffix.startswith("thousand"):
            val *= 1_000
        elif suffix.startswith("m") or suffix.startswith("million"):
            val *= 1_000_000
        elif suffix.startswith("b") or suffix.startswith("billion"):
            val *= 1_000_000_000
        elif suffix.startswith("t") or suffix.startswith("trillion"):
            val *= 1_000_000_000_000
        # Normalise direction
        if direction in (
            "above",
            "over",
            "more than",
            "at least",
            "exceed",
            "reach",
            "hit",
        ):
            direction = "above"
        else:
            direction = "below"
        entities.number_thresholds.append((direction, val))

    # --- Dates ---
    for m in _RE_DATE_MDY.finditer(text):
        month_str = m.group(1).lower()[:3]
        day = m.group(2)
        year = m.group(3) or ""
        entities.dates.add(f"{month_str}{day}{year}".strip())
    for m in _RE_DATE_YEAR.finditer(text):
        entities.dates.add(m.group(1))

    # --- Timeframes ---
    for m in _RE_DATE_BY.finditer(text):
        entities.timeframes.add(m.group(0).lower().strip())

    # --- Event types ---
    for keyword, event_type in _EVENT_KEYWORDS.items():
        if keyword in lower:
            entities.event_types.add(event_type)

    # --- Resolution methods ---
    for phrase, method in _RESOLUTION_KEYWORDS.items():
        if phrase in lower:
            entities.resolution_methods.add(method)

    # --- Person / organisation names (heuristic: capitalised sequences) ---
    for m in _RE_CAPITALIZED_SEQUENCE.finditer(text):
        candidate = m.group(1)
        words = candidate.split()
        # Filter out sequences that are entirely common title words
        if all(w in _COMMON_TITLE_WORDS for w in words):
            continue
        # If the candidate overlaps with a known team alias, skip it
        # (already captured under teams)
        if candidate.lower() in _ALIAS_TO_CANONICAL:
            continue
        # Treat as person or organisation
        entities.persons.add(candidate.lower())

    return entities


# ====================================================================== #
#  Entity overlap scoring
# ====================================================================== #


def _entity_overlap_score(e1: ExtractedEntities, e2: ExtractedEntities) -> float:
    """Compute a Jaccard-style overlap of extracted entities (0..1).

    Entities are compared by their string representations.  An overlap
    score of 1.0 means every entity found in one question was also found
    in the other.  0.0 means no overlap at all.
    """
    s1 = e1.all_entity_strings()
    s2 = e2.all_entity_strings()
    if not s1 and not s2:
        # Both sides have no extractable entities -- not a disqualifier
        return 1.0
    if not s1 or not s2:
        # One side has entities, the other does not -- weak signal
        return 0.3
    intersection = s1 & s2
    union = s1 | s2
    return len(intersection) / len(union) if union else 0.0


# ====================================================================== #
#  Resolution criteria validation
# ====================================================================== #


def _resolution_dates_compatible(
    poly_market: Market,
    kalshi_market: Market,
    tolerance_days: int = 3,
) -> bool:
    """Check whether two markets resolve at approximately the same time."""
    d1 = poly_market.end_date
    d2 = kalshi_market.end_date
    if d1 is None or d2 is None:
        # Cannot confirm or deny -- allow the match to proceed
        return True
    d1_naive = d1.replace(tzinfo=None) if d1.tzinfo else d1
    d2_naive = d2.replace(tzinfo=None) if d2.tzinfo else d2
    return abs(d1_naive - d2_naive) <= timedelta(days=tolerance_days)


def _resolution_methods_compatible(
    e1: ExtractedEntities, e2: ExtractedEntities
) -> tuple[bool, str]:
    """Check whether the resolution methods extracted from both markets
    are compatible.  Returns (is_compatible, reason_string).

    If one market resolves on *closing price* and the other on
    *any intraday touch*, they are NOT semantically equivalent even if
    the question text looks similar ("BTC above $100k by Friday").

    This addresses the Semantic Non-Fungibility problem described in
    academic literature on prediction market arbitrage.
    """
    m1 = e1.resolution_methods
    m2 = e2.resolution_methods

    if not m1 and not m2:
        return (
            True,
            "no explicit resolution method detected on either side (proceed with caution)",
        )

    if not m1 or not m2:
        # One side has explicit resolution criteria, the other doesn't.
        # This is a risk - they might resolve differently.
        return (
            True,
            "WARNING: resolution method detected on only one side; compatibility unconfirmed",
        )

    # Both sides have resolution methods -- check for conflicts
    if m1 == m2:
        return True, f"matching resolution methods: {m1}"

    # Specific known conflicts
    conflicting_pairs = [
        ({"closing_price"}, {"any_point"}),
        ({"end_of_day"}, {"any_point"}),
    ]
    for pair_a, pair_b in conflicting_pairs:
        if (m1 & pair_a and m2 & pair_b) or (m1 & pair_b and m2 & pair_a):
            return False, (
                f"resolution conflict: {m1} vs {m2} -- "
                f"'closing price' and 'any intraday point' are semantically different"
            )

    # Methods differ but are not a known hard conflict -- flag as warning
    return True, f"resolution methods differ ({m1} vs {m2}) but no known hard conflict"


# ====================================================================== #
#  Sports-specific matching helpers
# ====================================================================== #


def _normalize_team_name(name: str) -> Optional[str]:
    """Resolve a team name/abbreviation to its canonical form, or None."""
    return _ALIAS_TO_CANONICAL.get(name.lower().strip())


def _sports_entities_compatible(e1: ExtractedEntities, e2: ExtractedEntities) -> bool:
    """For sports markets, teams MUST match."""
    if not e1.teams and not e2.teams:
        return True  # not a sports market
    if not e1.teams or not e2.teams:
        return False  # one side has teams, the other does not
    return bool(e1.teams & e2.teams)


# ====================================================================== #
#  Sport outcome type detection (3-way market guard)
# ====================================================================== #

# Kalshi ticker suffixes that indicate the outcome type
_KALSHI_DRAW_SUFFIXES = ("-TIE", "-DRW", "-DRAW")
_KALSHI_HOME_SUFFIXES = ("-WIN", "-HOM", "-HOME")
_KALSHI_AWAY_SUFFIXES = ("-AWY", "-AWAY", "-VIS")

_DRAW_KEYWORDS = frozenset(
    {
        "tie",
        "draw",
        "drawn",
        "tied",
        "ties",
        "draws",
    }
)
_WIN_KEYWORDS = frozenset(
    {
        "win",
        "winner",
        "wins",
        "victory",
        "victorious",
        "beat",
        "beats",
        "defeat",
        "defeats",
    }
)


def _extract_sport_outcome_type(market: Market) -> Optional[str]:
    """Classify a market as 'draw' or 'team_win' based on ticker/question."""
    ticker_upper = market.id.upper()
    if any(ticker_upper.endswith(s) for s in _KALSHI_DRAW_SUFFIXES):
        return "draw"
    if any(ticker_upper.endswith(s) for s in _KALSHI_HOME_SUFFIXES):
        return "team_win"
    if any(ticker_upper.endswith(s) for s in _KALSHI_AWAY_SUFFIXES):
        return "team_win"

    q = market.question.lower()
    has_draw = any(kw in q for kw in _DRAW_KEYWORDS)
    has_win = any(kw in q for kw in _WIN_KEYWORDS)
    if has_draw and not has_win:
        return "draw"
    if has_win and not has_draw:
        return "team_win"
    return None


def _sport_outcome_types_compatible(poly_market: Market, kalshi_market: Market) -> bool:
    """Return True if both markets have compatible sport outcome types.

    Prevents matching a 'Team Win' market against a 'Draw' market in
    3-way soccer events — the most common source of catastrophic false
    positives in cross-platform sports arbitrage.
    """
    pm_type = _extract_sport_outcome_type(poly_market)
    k_type = _extract_sport_outcome_type(kalshi_market)
    if pm_type is None or k_type is None:
        return True  # can't determine — allow the match
    return pm_type == k_type


# ====================================================================== #
#  Crypto-specific matching helpers
# ====================================================================== #


def _crypto_entities_compatible(e1: ExtractedEntities, e2: ExtractedEntities) -> bool:
    """For crypto markets, assets and price targets must align."""
    if not e1.crypto_assets and not e2.crypto_assets:
        return True  # not a crypto market
    if not e1.crypto_assets or not e2.crypto_assets:
        return False

    # Assets must overlap
    if not (e1.crypto_assets & e2.crypto_assets):
        return False

    # If both have price targets, they must be close (within 2%)
    if e1.price_targets and e2.price_targets:
        # Compare all pairs; at least one pair must be within tolerance
        for p1 in e1.price_targets:
            for p2 in e2.price_targets:
                if p1 == 0 and p2 == 0:
                    continue
                mid = (p1 + p2) / 2.0
                if mid > 0 and abs(p1 - p2) / mid <= 0.02:
                    return True
        return False

    return True


# ====================================================================== #
#  MarketMatcher: 4-stage matching pipeline
# ====================================================================== #


class MarketMatcher:
    """Implements the multi-stage matching pipeline for cross-platform
    market comparison.

    Stage 1: Text similarity filter (fast, eliminates most non-matches)
    Stage 2: Entity overlap check (must share key entities)
    Stage 3: Timeframe / resolution validation (must resolve similarly)
    Stage 4: Confidence scoring (combined score from all stages)

    Usage::

        matcher = MarketMatcher(text_threshold=0.75)
        result = matcher.match(poly_market, kalshi_market)
        if result is not None:
            print(result.overall_confidence, result.match_reasoning)
    """

    def __init__(
        self,
        text_threshold: float = 0.75,
        entity_threshold: float = 0.30,
        confidence_threshold: float = 0.55,
        # Weights for the final confidence score
        w_text: float = 0.40,
        w_entity: float = 0.35,
        w_resolution: float = 0.25,
    ):
        self.text_threshold = text_threshold
        self.entity_threshold = entity_threshold
        self.confidence_threshold = confidence_threshold
        self.w_text = w_text
        self.w_entity = w_entity
        self.w_resolution = w_resolution

    # -------------------------------------------------------------- #
    #  Public API
    # -------------------------------------------------------------- #

    def match(
        self,
        poly_market: Market,
        kalshi_market: Market,
    ) -> Optional[MatchResult]:
        """Run the 4-stage pipeline and return a MatchResult if the markets
        are considered equivalent, or None if they fail any gate."""

        reasoning_parts: list[str] = []

        # ---- Stage 1: Text similarity ----
        text_sim = _fuzzy_similarity(poly_market.question, kalshi_market.question)
        if text_sim < self.text_threshold:
            return None  # fast reject
        reasoning_parts.append(f"text_similarity={text_sim:.3f}")

        # ---- Stage 2: Entity overlap ----
        e_poly = _extract_entities(poly_market.question)
        e_kalshi = _extract_entities(kalshi_market.question)

        entity_score = _entity_overlap_score(e_poly, e_kalshi)
        if entity_score < self.entity_threshold:
            return None
        reasoning_parts.append(f"entity_overlap={entity_score:.3f}")

        # Domain-specific gates
        if not _sports_entities_compatible(e_poly, e_kalshi):
            return None
        if not _crypto_entities_compatible(e_poly, e_kalshi):
            return None

        # 3-way sport outcome gate — never match WIN vs TIE/DRAW
        if not _sport_outcome_types_compatible(poly_market, kalshi_market):
            return None

        # ---- Stage 3: Resolution validation ----
        dates_ok = _resolution_dates_compatible(poly_market, kalshi_market)
        methods_ok, method_reason = _resolution_methods_compatible(e_poly, e_kalshi)
        resolution_match = dates_ok and methods_ok
        reasoning_parts.append(f"resolution_match={resolution_match} ({method_reason})")

        # If resolution dates are incompatible, hard reject
        if not dates_ok:
            return None

        # ---- Stage 4: Confidence scoring ----
        resolution_score = 1.0 if resolution_match else 0.4
        # Penalize when resolution compatibility is uncertain
        if resolution_match and "unconfirmed" in method_reason.lower():
            resolution_score = 0.7
        elif resolution_match and "caution" in method_reason.lower():
            resolution_score = 0.85
        overall = (
            self.w_text * text_sim
            + self.w_entity * entity_score
            + self.w_resolution * resolution_score
        )
        reasoning_parts.append(f"overall_confidence={overall:.3f}")

        if overall < self.confidence_threshold:
            return None

        # Compute entity sets for the result
        all_poly = e_poly.all_entity_strings()
        all_kalshi = e_kalshi.all_entity_strings()

        return MatchResult(
            poly_market=poly_market,
            kalshi_market=kalshi_market,
            text_similarity=round(text_sim, 4),
            entity_overlap=round(entity_score, 4),
            resolution_match=resolution_match,
            overall_confidence=round(overall, 4),
            match_reasoning="; ".join(reasoning_parts),
            shared_entities=sorted(all_poly & all_kalshi),
            poly_only_entities=sorted(all_poly - all_kalshi),
            kalshi_only_entities=sorted(all_kalshi - all_poly),
        )

    def match_events(
        self,
        poly_event: Event,
        kalshi_event: Event,
    ) -> list[MatchResult]:
        """Match all market pairs between two events and return the
        successful matches."""
        results: list[MatchResult] = []
        for p_mkt in poly_event.markets:
            for k_mkt in kalshi_event.markets:
                result = self.match(p_mkt, k_mkt)
                if result is not None:
                    results.append(result)
        return results


class CrossPlatformScanner:
    """Detect arbitrage opportunities across Polymarket and Kalshi.

    The scanner pulls events from both platforms, matches them using a
    multi-stage pipeline (fuzzy text similarity, entity extraction,
    resolution criteria validation), then compares prices to identify
    cross-platform arbitrage.

    The matching pipeline addresses the *Semantic Non-Fungibility*
    problem identified in academic literature: naive keyword overlap
    produces false arbitrage signals from incorrectly matched markets
    that look textually similar but differ in resolution semantics
    (e.g. "closing price" vs "any intraday point").

    An arbitrage exists when the sum of the cheapest YES on one platform
    and the cheapest NO on the other platform is less than $1.00 (after
    accounting for fees on both sides).
    """

    def __init__(
        self,
        *,
        text_threshold: float = 0.75,
        entity_threshold: float = 0.30,
        confidence_threshold: float = 0.55,
    ):
        self._poly_client = polymarket_client
        self._kalshi_client = kalshi_client
        self._matcher = MarketMatcher(
            text_threshold=text_threshold,
            entity_threshold=entity_threshold,
            confidence_threshold=confidence_threshold,
        )

    # ------------------------------------------------------------------ #
    #  Text normalisation and similarity (legacy helpers, still used
    #  for event-level pre-filtering before the full pipeline runs)
    # ------------------------------------------------------------------ #

    _STOP_WORDS = frozenset(
        {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "will",
            "be",
            "to",
            "of",
            "in",
            "on",
            "at",
            "by",
            "for",
            "and",
            "or",
            "not",
            "with",
            "it",
            "this",
            "that",
            "from",
            "as",
            "if",
            "do",
            "does",
            "did",
            "has",
            "have",
            "had",
            "but",
            "so",
            "than",
            "when",
            "what",
            "which",
            "who",
            "whom",
            "how",
            "no",
            "yes",
            "before",
            "after",
            "between",
        }
    )

    @classmethod
    def _tokenize(cls, text: str) -> set[str]:
        """Lower-case, strip punctuation, remove stop words."""
        words = re.findall(r"[a-z0-9]+", text.lower())
        return {w for w in words if w not in cls._STOP_WORDS and len(w) > 1}

    @classmethod
    def _title_similarity(cls, a: str, b: str) -> float:
        """Keyword-overlap Jaccard similarity in [0, 1]."""
        tokens_a = cls._tokenize(a)
        tokens_b = cls._tokenize(b)
        if not tokens_a or not tokens_b:
            return 0.0
        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b
        return len(intersection) / len(union)

    # ------------------------------------------------------------------ #
    #  Date proximity check
    # ------------------------------------------------------------------ #

    @staticmethod
    def _dates_close(
        d1: Optional[datetime],
        d2: Optional[datetime],
        tolerance_days: int = 3,
    ) -> bool:
        """Return True if both dates exist and are within *tolerance_days*."""
        if d1 is None or d2 is None:
            # If either side has no date, we cannot disqualify the match,
            # so treat as acceptable.
            return True
        # Ensure both are naive UTC for a fair comparison
        d1_naive = d1.replace(tzinfo=None) if d1.tzinfo else d1
        d2_naive = d2.replace(tzinfo=None) if d2.tzinfo else d2
        return abs(d1_naive - d2_naive) <= timedelta(days=tolerance_days)

    # ------------------------------------------------------------------ #
    #  Category matching
    # ------------------------------------------------------------------ #

    @staticmethod
    def _categories_compatible(cat_a: Optional[str], cat_b: Optional[str]) -> bool:
        """Fuzzy category match (case-insensitive substring)."""
        if cat_a is None or cat_b is None:
            return True  # unknown category is not a disqualifier
        a = cat_a.lower().strip()
        b = cat_b.lower().strip()
        if a == b:
            return True
        # Allow substring containment (e.g. "politics" in "US Politics")
        return a in b or b in a

    # ------------------------------------------------------------------ #
    #  Event matching (Stage 0 -- coarse pre-filter)
    # ------------------------------------------------------------------ #

    async def find_matching_events(
        self,
        similarity_threshold: float = 0.35,
    ) -> list[tuple[Event, Event]]:
        """Find pairs of (Polymarket event, Kalshi event) that *might*
        represent the same real-world question.

        This is a **coarse pre-filter** (Stage 0).  It uses cheap Jaccard
        keyword overlap on event titles to quickly eliminate obviously
        unrelated events before the expensive per-market matching
        pipeline runs.

        The threshold is intentionally kept LOW (default 0.35) because
        the downstream MarketMatcher pipeline applies much stricter
        fuzzy similarity, entity overlap, and resolution validation on
        the individual market pairs.

        Matching criteria:
            1. Title similarity >= *similarity_threshold* (Jaccard on keywords)
            2. Resolution dates within 3 days (when both are known)
            3. Categories are compatible
        """
        # Fetch events from both platforms concurrently
        try:
            poly_events, kalshi_events = await asyncio.gather(
                self._poly_client.get_all_events(closed=False),
                self._kalshi_client.get_all_events(closed=False),
            )
        except Exception as exc:
            logger.error("Failed to fetch events for matching", error=str(exc))
            return []

        logger.info(
            "Loaded events for matching",
            polymarket_count=len(poly_events),
            kalshi_count=len(kalshi_events),
        )

        if not poly_events or not kalshi_events:
            return []

        matched: list[tuple[Event, Event]] = []

        for p_event in poly_events:
            best_score = 0.0
            best_kalshi: Optional[Event] = None

            for k_event in kalshi_events:
                # Quick category gate
                if not self._categories_compatible(p_event.category, k_event.category):
                    continue

                score = self._title_similarity(p_event.title, k_event.title)
                if score < similarity_threshold:
                    continue

                # Check resolution dates across markets
                p_end = _earliest_end_date(p_event)
                k_end = _earliest_end_date(k_event)
                if not self._dates_close(p_end, k_end):
                    continue

                if score > best_score:
                    best_score = score
                    best_kalshi = k_event

            if best_kalshi is not None:
                matched.append((p_event, best_kalshi))
                logger.debug(
                    "Matched cross-platform event (coarse)",
                    poly=p_event.title[:60],
                    kalshi=best_kalshi.title[:60],
                    score=round(best_score, 3),
                )

        logger.info("Cross-platform coarse event matches", count=len(matched))
        return matched

    # ------------------------------------------------------------------ #
    #  Arbitrage calculation for a single market pair
    # ------------------------------------------------------------------ #

    @staticmethod
    def calculate_cross_platform_arb(
        poly_market: Market,
        kalshi_market: Market,
        poly_fee: float = POLYMARKET_FEE,
        kalshi_fee: float = KALSHI_FEE,
        match_result: Optional[MatchResult] = None,
    ) -> Optional[ArbitrageOpportunity]:
        """Detect whether buying YES on one platform and NO on the other
        yields a risk-free profit after fees.

        Two legs to check:
            Leg A: buy Polymarket YES  +  buy Kalshi NO
            Leg B: buy Kalshi YES      +  buy Polymarket NO

        For each leg the guaranteed payout is $1.00 (one side always wins).
        Profit = payout - fees_on_winning_side - total_cost.

        If a *match_result* from the MarketMatcher pipeline is provided,
        it is used to enrich the opportunity with match-confidence
        metadata and to adjust the risk score based on match quality.
        """
        p_yes = poly_market.yes_price
        p_no = poly_market.no_price
        k_yes = kalshi_market.yes_price
        k_no = kalshi_market.no_price

        # Need valid prices on both sides
        if not (0 < p_yes < 1 and 0 < p_no < 1):
            return None
        if not (0 < k_yes < 1 and 0 < k_no < 1):
            return None

        # Adjust Kalshi fee based on market volume if available
        if hasattr(kalshi_market, "volume") and kalshi_market.volume > 0:
            kalshi_fee = kalshi_fee_for_volume(kalshi_market.volume)

        best_opportunity: Optional[ArbitrageOpportunity] = None
        best_net = 0.0

        legs = [
            # (description, cost_platform_a, cost_platform_b, fee_on_a_win, fee_on_b_win, label)
            (
                "Buy YES on Polymarket + NO on Kalshi",
                p_yes,  # cost if Poly YES wins
                k_no,  # cost if Kalshi NO wins
                poly_fee,
                kalshi_fee,
                "poly_yes_kalshi_no",
            ),
            (
                "Buy NO on Polymarket + YES on Kalshi",
                p_no,  # cost if Poly NO wins
                k_yes,  # cost if Kalshi YES wins
                poly_fee,
                kalshi_fee,
                "poly_no_kalshi_yes",
            ),
        ]

        for desc, cost_a, cost_b, fee_a, fee_b, label in legs:
            total_cost = cost_a + cost_b
            if total_cost >= 1.0:
                continue  # no arb

            # When the first leg wins, the payout is $1 on that platform
            # minus the fee on the winning amount.
            # The *winning* amount is (1 - cost_of_that_side).
            # Worst-case fee is max of the two scenarios.
            profit_if_a_wins = (1.0 - cost_a) * (1.0 - fee_a) - cost_b
            profit_if_b_wins = (1.0 - cost_b) * (1.0 - fee_b) - cost_a

            # Guaranteed profit is the *minimum* of the two scenarios
            guaranteed = min(profit_if_a_wins, profit_if_b_wins)
            if guaranteed <= 0:
                continue

            gross = 1.0 - total_cost
            fee_estimate = gross - guaranteed
            roi = (guaranteed / total_cost) * 100.0 if total_cost > 0 else 0.0

            if guaranteed > best_net:
                best_net = guaranteed

                # Build the positions-to-take list
                if label == "poly_yes_kalshi_no":
                    positions = [
                        {
                            "platform": "polymarket",
                            "market_id": poly_market.id,
                            "side": "YES",
                            "price": p_yes,
                        },
                        {
                            "platform": "kalshi",
                            "market_id": kalshi_market.id,
                            "side": "NO",
                            "price": k_no,
                        },
                    ]
                else:
                    positions = [
                        {
                            "platform": "polymarket",
                            "market_id": poly_market.id,
                            "side": "NO",
                            "price": p_no,
                        },
                        {
                            "platform": "kalshi",
                            "market_id": kalshi_market.id,
                            "side": "YES",
                            "price": k_yes,
                        },
                    ]

                # Compute risk score: base is low (0.15) for guaranteed
                # arbs, but increase if match confidence is mediocre or
                # resolution criteria differ.
                risk_score = 0.15
                risk_factors_list = _risk_factors(poly_market, kalshi_market)

                if match_result is not None:
                    # Scale risk inversely with match confidence
                    # confidence 1.0 -> no extra risk
                    # confidence 0.55 (threshold) -> +0.30 risk
                    confidence_penalty = max(
                        0.0, 0.30 * (1.0 - match_result.overall_confidence)
                    )
                    risk_score = min(1.0, risk_score + confidence_penalty)

                    if not match_result.resolution_match:
                        risk_score = min(1.0, risk_score + 0.20)
                        risk_factors_list.append(
                            "Resolution criteria may differ between platforms"
                        )

                    if match_result.poly_only_entities:
                        risk_factors_list.append(
                            f"Entities only on Polymarket: "
                            f"{', '.join(match_result.poly_only_entities[:5])}"
                        )
                    if match_result.kalshi_only_entities:
                        risk_factors_list.append(
                            f"Entities only on Kalshi: "
                            f"{', '.join(match_result.kalshi_only_entities[:5])}"
                        )

                # Build enriched description
                description = (
                    f"{desc}. "
                    f"Poly YES={p_yes:.3f} NO={p_no:.3f} | "
                    f"Kalshi YES={k_yes:.3f} NO={k_no:.3f}"
                )
                if match_result is not None:
                    description += (
                        f" | Match confidence={match_result.overall_confidence:.2f}"
                        f" (text={match_result.text_similarity:.2f},"
                        f" entities={match_result.entity_overlap:.2f},"
                        f" resolution={'OK' if match_result.resolution_match else 'WARN'})"
                    )

                # Market metadata dicts with match info
                poly_market_dict: dict = {
                    "id": poly_market.id,
                    "platform": "polymarket",
                    "question": poly_market.question,
                    "yes_price": p_yes,
                    "no_price": p_no,
                }
                kalshi_market_dict: dict = {
                    "id": kalshi_market.id,
                    "platform": "kalshi",
                    "question": kalshi_market.question,
                    "yes_price": k_yes,
                    "no_price": k_no,
                }
                if match_result is not None:
                    poly_market_dict["match_confidence"] = (
                        match_result.overall_confidence
                    )
                    kalshi_market_dict["match_confidence"] = (
                        match_result.overall_confidence
                    )
                    poly_market_dict["match_reasoning"] = match_result.match_reasoning
                    kalshi_market_dict["match_reasoning"] = match_result.match_reasoning

                best_opportunity = ArbitrageOpportunity(
                    strategy=StrategyType.CROSS_PLATFORM,
                    title=f"Cross-platform arb: {poly_market.question[:80]}",
                    description=description,
                    total_cost=round(total_cost, 6),
                    expected_payout=1.0,
                    gross_profit=round(gross, 6),
                    fee=round(fee_estimate, 6),
                    net_profit=round(guaranteed, 6),
                    roi_percent=round(roi, 4),
                    risk_score=round(risk_score, 4),
                    risk_factors=risk_factors_list,
                    markets=[poly_market_dict, kalshi_market_dict],
                    category=None,
                    min_liquidity=min(poly_market.liquidity, kalshi_market.liquidity),
                    positions_to_take=positions,
                    resolution_date=poly_market.end_date or kalshi_market.end_date,
                )

        return best_opportunity

    # ------------------------------------------------------------------ #
    #  Full scan (enhanced with MarketMatcher pipeline)
    # ------------------------------------------------------------------ #

    async def scan_cross_platform(
        self,
        similarity_threshold: float = 0.35,
    ) -> list[ArbitrageOpportunity]:
        """Run the full cross-platform arbitrage scan.

        Steps:
            1. Coarse event matching (Jaccard keyword overlap on titles).
            2. For every matched event pair, run the MarketMatcher
               pipeline on each (Polymarket market, Kalshi market) pair
               to validate semantic equivalence via fuzzy text similarity,
               entity extraction, and resolution criteria validation.
            3. Only for validated market pairs, compute arbitrage.
            4. Return all detected opportunities sorted by ROI descending.

        The multi-stage pipeline dramatically reduces false positives
        compared to the previous simple keyword-overlap approach.
        """
        logger.info(
            "Starting cross-platform arbitrage scan (enhanced matching)",
            text_threshold=self._matcher.text_threshold,
            entity_threshold=self._matcher.entity_threshold,
            confidence_threshold=self._matcher.confidence_threshold,
        )

        matches = await self.find_matching_events(
            similarity_threshold=similarity_threshold,
        )
        if not matches:
            logger.info("No cross-platform matches found")
            return []

        opportunities: list[ArbitrageOpportunity] = []
        total_pairs_checked = 0
        total_pairs_matched = 0

        for poly_event, kalshi_event in matches:
            poly_markets = poly_event.markets
            kalshi_markets = kalshi_event.markets

            # If either event has no inline markets, skip.
            if not poly_markets or not kalshi_markets:
                continue

            # Run the MarketMatcher pipeline on all market pairs
            # within this event pair
            for p_mkt in poly_markets:
                for k_mkt in kalshi_markets:
                    total_pairs_checked += 1

                    # Run the 4-stage matching pipeline
                    match_result = self._matcher.match(p_mkt, k_mkt)

                    if match_result is None:
                        # Markets did not pass the matching pipeline --
                        # skip to avoid false arbitrage signals.
                        continue

                    total_pairs_matched += 1

                    logger.debug(
                        "Market pair passed matching pipeline",
                        poly_q=p_mkt.question[:60],
                        kalshi_q=k_mkt.question[:60],
                        confidence=match_result.overall_confidence,
                        text_sim=match_result.text_similarity,
                        entity_overlap=match_result.entity_overlap,
                        resolution_match=match_result.resolution_match,
                    )

                    opp = self.calculate_cross_platform_arb(
                        p_mkt,
                        k_mkt,
                        match_result=match_result,
                    )
                    if opp is not None:
                        opp.event_id = poly_event.id
                        opp.event_title = poly_event.title
                        opp.category = poly_event.category or kalshi_event.category
                        opportunities.append(opp)

        # Highest ROI first
        opportunities.sort(key=lambda o: o.roi_percent, reverse=True)
        logger.info(
            "Cross-platform scan complete",
            market_pairs_checked=total_pairs_checked,
            market_pairs_matched=total_pairs_matched,
            opportunities=len(opportunities),
        )
        return opportunities


# ------------------------------------------------------------------ #
#  Module-level helpers
# ------------------------------------------------------------------ #


def _earliest_end_date(event: Event) -> Optional[datetime]:
    """Return the earliest end_date across all markets in an event."""
    dates = [m.end_date for m in event.markets if m.end_date is not None]
    return min(dates) if dates else None


def _risk_factors(poly_market: Market, kalshi_market: Market) -> list[str]:
    """Compile a list of risk factors for the opportunity."""
    factors: list[str] = []
    if poly_market.liquidity < 1000:
        factors.append("Low Polymarket liquidity")
    if kalshi_market.liquidity < 1000:
        factors.append("Low Kalshi liquidity")
    if poly_market.volume < 500:
        factors.append("Low Polymarket volume")
    if kalshi_market.volume < 500:
        factors.append("Low Kalshi volume")
    factors.append("Execution on two separate platforms required")
    factors.append("Settlement timing may differ across platforms")
    return factors


# Singleton instance
cross_platform_scanner = CrossPlatformScanner()
