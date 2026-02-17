"""
LLM Reranker -- Cuts false positives before expensive edge estimation.

Takes top-K retrieval candidates and calls an LLM to score relevance
of each (article, market) pair. Returns top-N with relevance score
and brief rationale.

Market-type-aware prompting: detects domain (sports, politics, crypto,
entertainment) from the market question/slug and selects a tailored
system prompt that emphasizes domain-relevant signals.

Includes pre-LLM entity-type overlap filtering to reject clearly
mismatched pairs (e.g. baseball article vs basketball market) before
spending LLM budget.

Pattern from: Polymarket Agents (RAG reranking pipeline).
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from services.news.hybrid_retriever import RetrievalCandidate

logger = logging.getLogger(__name__)

_LLM_CALL_TIMEOUT_SECONDS = 20.0

# ---------------------------------------------------------------------------
# Entity-type overlap filtering (pre-LLM, regex-based NER)
# ---------------------------------------------------------------------------
# Minimum entity overlap score below which an article-market pair is heavily
# penalised.  Pairs with clearly mismatched entity *types* (e.g. a baseball
# team in the article vs. a political figure in the market) are penalised
# even more aggressively.
ENTITY_OVERLAP_MIN_SCORE = 0.10
ENTITY_MISMATCH_PENALTY = 0.15  # subtracted from rerank_score for type clashes

# -- Sports teams (major North American + European leagues) -----------------
_NBA_TEAMS = {
    "hawks", "celtics", "nets", "hornets", "bulls", "cavaliers", "mavericks",
    "nuggets", "pistons", "warriors", "rockets", "pacers", "clippers", "lakers",
    "grizzlies", "heat", "bucks", "timberwolves", "pelicans", "knicks",
    "thunder", "magic", "76ers", "sixers", "suns", "blazers", "kings",
    "spurs", "raptors", "jazz", "wizards",
}
_NFL_TEAMS = {
    "cardinals", "falcons", "ravens", "bills", "panthers", "bears",
    "bengals", "browns", "cowboys", "broncos", "lions", "packers", "texans",
    "colts", "jaguars", "chiefs", "chargers", "rams", "dolphins", "vikings",
    "patriots", "saints", "giants", "jets", "raiders", "eagles", "steelers",
    "49ers", "seahawks", "buccaneers", "commanders", "titans",
}
_MLB_TEAMS = {
    "diamondbacks", "braves", "orioles", "red sox", "cubs", "white sox",
    "reds", "guardians", "rockies", "tigers", "astros", "royals", "angels",
    "dodgers", "marlins", "brewers", "twins", "mets", "yankees", "athletics",
    "phillies", "pirates", "padres", "mariners", "cardinals", "rays",
    "rangers", "blue jays", "nationals",
}
_NHL_TEAMS = {
    "ducks", "coyotes", "bruins", "sabres", "flames", "hurricanes",
    "blackhawks", "avalanche", "blue jackets", "stars", "red wings",
    "oilers", "panthers", "kings", "wild", "canadiens", "predators",
    "devils", "islanders", "rangers", "senators", "flyers", "penguins",
    "sharks", "kraken", "blues", "lightning", "maple leafs", "canucks",
    "golden knights", "capitals", "jets",
}
_SOCCER_TEAMS = {
    "arsenal", "chelsea", "liverpool", "manchester united", "manchester city",
    "tottenham", "barcelona", "real madrid", "bayern munich", "juventus",
    "psg", "inter milan", "ac milan", "borussia dortmund", "atletico madrid",
}

_ALL_SPORTS_TEAMS = (
    _NBA_TEAMS | _NFL_TEAMS | _MLB_TEAMS | _NHL_TEAMS | _SOCCER_TEAMS
)
_SPORTS_KEYWORDS = {
    "nba", "nfl", "mlb", "nhl", "mls", "epl", "premier league", "la liga",
    "serie a", "bundesliga", "champions league", "playoffs", "super bowl",
    "world series", "stanley cup", "finals", "championship", "touchdown",
    "home run", "goal", "assist", "quarterback", "pitcher", "goalie",
    "halftime", "overtime", "inning", "roster", "draft", "trade deadline",
    "free agency", "mvp", "coach", "manager",
}

# -- Political figures / government -----------------------------------------
_POLITICAL_KEYWORDS = {
    "president", "senator", "congressman", "congresswoman", "governor",
    "prime minister", "chancellor", "parliament", "congress", "senate",
    "house of representatives", "white house", "cabinet", "secretary of state",
    "supreme court", "justice", "democrat", "republican", "gop",
    "election", "vote", "ballot", "poll", "campaign", "inauguration",
    "executive order", "bill", "legislation", "veto", "impeach",
}

# -- Company / business -----------------------------------------------------
_COMPANY_KEYWORDS = {
    "ceo", "cfo", "cto", "ipo", "earnings", "revenue", "stock", "shares",
    "market cap", "quarterly", "dividend", "profit", "merger", "acquisition",
    "startup", "valuation", "nasdaq", "nyse", "s&p", "dow jones",
}

# -- Crypto / blockchain ----------------------------------------------------
_CRYPTO_KEYWORDS = {
    "bitcoin", "btc", "ethereum", "eth", "crypto", "blockchain", "defi",
    "token", "altcoin", "mining", "wallet", "exchange", "binance", "coinbase",
    "solana", "sol", "cardano", "ada", "dogecoin", "doge", "nft",
}

# -- Country / geopolitical -------------------------------------------------
_COUNTRIES = {
    "united states", "usa", "us", "china", "russia", "ukraine", "india",
    "brazil", "uk", "united kingdom", "france", "germany", "japan",
    "south korea", "north korea", "iran", "israel", "palestine", "taiwan",
    "mexico", "canada", "australia", "saudi arabia", "turkey", "egypt",
    "pakistan", "indonesia", "nigeria", "south africa",
}

# Entity type enum
ENTITY_TYPE_SPORTS = "sports"
ENTITY_TYPE_POLITICS = "politics"
ENTITY_TYPE_BUSINESS = "business"
ENTITY_TYPE_CRYPTO = "crypto"
ENTITY_TYPE_GEO = "geopolitical"
ENTITY_TYPE_OTHER = "other"

# Proper-noun regex (multi-word capitalised phrases)
_PROPER_NOUN_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")


# Keywords that are short enough to cause false-positive substring matches
# (e.g. "nfl" inside "inflation", "eth" inside "method") need word-boundary
# matching.  We pre-compile regex patterns for all keywords <= 4 chars and
# multi-word phrases.  Longer single words are safe for plain ``in`` checks.
_WORD_BOUNDARY_CACHE: Dict[str, re.Pattern] = {}


def _keyword_in_text(keyword: str, text_lower: str) -> bool:
    """Check if *keyword* appears in *text_lower* as a whole word/phrase.

    For short keywords (<=4 chars) or keywords that are common substrings
    of longer words, uses word-boundary regex to avoid false positives
    like 'nfl' matching inside 'inflation'.
    """
    # Multi-word phrases: substring match is fine (very unlikely to be
    # accidental substrings of other phrases).
    if " " in keyword and len(keyword) > 6:
        return keyword in text_lower

    # Short single-token keywords need word-boundary matching.
    if len(keyword) <= 5:
        pat = _WORD_BOUNDARY_CACHE.get(keyword)
        if pat is None:
            pat = re.compile(r"\b" + re.escape(keyword) + r"\b")
            _WORD_BOUNDARY_CACHE[keyword] = pat
        return bool(pat.search(text_lower))

    # Longer keywords: plain substring is fine.
    return keyword in text_lower


# Sports league tags for intra-sports mismatch detection
_LEAGUE_TAG_MAP: Dict[str, str] = {}
for _team in _NBA_TEAMS:
    _LEAGUE_TAG_MAP[_team] = "nba"
for _team in _NFL_TEAMS:
    _LEAGUE_TAG_MAP[_team] = "nfl"
for _team in _MLB_TEAMS:
    _LEAGUE_TAG_MAP[_team] = "mlb"
for _team in _NHL_TEAMS:
    _LEAGUE_TAG_MAP[_team] = "nhl"
for _team in _SOCCER_TEAMS:
    _LEAGUE_TAG_MAP[_team] = "soccer"
# Explicit league keywords
_LEAGUE_TAG_MAP.update({
    "nba": "nba", "nfl": "nfl", "mlb": "mlb", "nhl": "nhl",
    "mls": "soccer", "epl": "soccer", "premier league": "soccer",
    "la liga": "soccer", "serie a": "soccer", "bundesliga": "soccer",
    "champions league": "soccer",
})


def _extract_entities(text: str) -> Dict[str, Any]:
    """Extract named entities and classify entity types from text.

    Uses word-boundary-aware matching for short keywords to prevent
    false positives (e.g. "nfl" inside "inflation").

    Returns a dict with:
      - entities: list of extracted entity strings
      - types: set of entity-type labels detected
      - sports_entities: set of sports team / league tokens found
      - sports_leagues: set of league tags (nba, nfl, mlb, nhl, soccer)
      - political_entities: set of political tokens found
      - company_entities: set of business tokens found
      - crypto_entities: set of crypto tokens found
      - geo_entities: set of country/region tokens found
    """
    text_lower = text.lower()
    entities: List[str] = []
    types: Set[str] = set()

    sports_ents: Set[str] = set()
    sports_leagues: Set[str] = set()
    political_ents: Set[str] = set()
    company_ents: Set[str] = set()
    crypto_ents: Set[str] = set()
    geo_ents: Set[str] = set()

    # Proper nouns
    proper_nouns = _PROPER_NOUN_RE.findall(text)
    entities.extend(proper_nouns)

    # Sports teams (word-boundary aware)
    for team in _ALL_SPORTS_TEAMS:
        if _keyword_in_text(team, text_lower):
            sports_ents.add(team)
            types.add(ENTITY_TYPE_SPORTS)
            league = _LEAGUE_TAG_MAP.get(team)
            if league:
                sports_leagues.add(league)
    for kw in _SPORTS_KEYWORDS:
        if _keyword_in_text(kw, text_lower):
            sports_ents.add(kw)
            types.add(ENTITY_TYPE_SPORTS)
            league = _LEAGUE_TAG_MAP.get(kw)
            if league:
                sports_leagues.add(league)

    # Political (word-boundary aware)
    for kw in _POLITICAL_KEYWORDS:
        if _keyword_in_text(kw, text_lower):
            political_ents.add(kw)
            types.add(ENTITY_TYPE_POLITICS)

    # Business (word-boundary aware)
    for kw in _COMPANY_KEYWORDS:
        if _keyword_in_text(kw, text_lower):
            company_ents.add(kw)
            types.add(ENTITY_TYPE_BUSINESS)

    # Crypto (word-boundary aware)
    for kw in _CRYPTO_KEYWORDS:
        if _keyword_in_text(kw, text_lower):
            crypto_ents.add(kw)
            types.add(ENTITY_TYPE_CRYPTO)

    # Countries (word-boundary aware for short ones like "us", "uk")
    for country in _COUNTRIES:
        if _keyword_in_text(country, text_lower):
            geo_ents.add(country)
            types.add(ENTITY_TYPE_GEO)

    return {
        "entities": entities,
        "types": types,
        "sports_entities": sports_ents,
        "sports_leagues": sports_leagues,
        "political_entities": political_ents,
        "company_entities": company_ents,
        "crypto_entities": crypto_ents,
        "geo_entities": geo_ents,
    }


def _compute_entity_overlap(
    article_entities: Dict[str, Any],
    market_entities: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute entity overlap between article and market.

    Returns a dict with:
      - entity_overlap_score: float 0-1, overall overlap quality
      - type_match: bool, whether dominant entity types overlap
      - type_mismatch: bool, whether entity types clearly conflict
      - shared_entities: list of entity strings found in both
      - shared_types: set of shared entity type labels
      - article_types: set of article entity type labels
      - market_types: set of market entity type labels
      - detail: str, human-readable explanation
    """
    a_types = article_entities["types"]
    m_types = market_entities["types"]
    shared_types = a_types & m_types

    # Gather all typed entity sets and compute overlap per type
    typed_overlaps: Dict[str, Set[str]] = {}
    for etype, key in [
        (ENTITY_TYPE_SPORTS, "sports_entities"),
        (ENTITY_TYPE_POLITICS, "political_entities"),
        (ENTITY_TYPE_BUSINESS, "company_entities"),
        (ENTITY_TYPE_CRYPTO, "crypto_entities"),
        (ENTITY_TYPE_GEO, "geo_entities"),
    ]:
        a_set = article_entities.get(key, set())
        m_set = market_entities.get(key, set())
        overlap = a_set & m_set
        if overlap:
            typed_overlaps[etype] = overlap

    # Proper noun overlap (case-insensitive)
    a_nouns = {n.lower() for n in article_entities.get("entities", [])}
    m_nouns = {n.lower() for n in market_entities.get("entities", [])}
    shared_nouns = a_nouns & m_nouns

    # Build a combined shared entity list
    all_shared: Set[str] = set(shared_nouns)
    for overlap_set in typed_overlaps.values():
        all_shared.update(overlap_set)

    # Score computation
    # Base: weighted mix of type overlap and entity overlap
    if not a_types and not m_types:
        # Neither has strong typed entities -- fall back to proper noun overlap
        if not a_nouns and not m_nouns:
            score = 0.5  # neutral, no signal
            detail = "No typed entities in either text; neutral overlap."
        elif a_nouns and m_nouns:
            noun_ratio = len(shared_nouns) / max(1, min(len(a_nouns), len(m_nouns)))
            score = 0.3 + 0.7 * min(1.0, noun_ratio)
            detail = f"Proper noun overlap: {len(shared_nouns)} shared of {len(a_nouns)}a/{len(m_nouns)}m."
        else:
            score = 0.4
            detail = "Only one side has proper nouns; weak signal."
    else:
        # At least one side has typed entities
        type_score = len(shared_types) / max(1, min(len(a_types), len(m_types))) if (a_types and m_types) else 0.0
        entity_score = len(all_shared) / max(1, min(len(a_nouns) + sum(len(v) for v in article_entities.values() if isinstance(v, set)), 1)) if all_shared else 0.0
        entity_score = min(1.0, entity_score)
        score = 0.5 * type_score + 0.3 * min(1.0, entity_score) + 0.2 * (1.0 if shared_nouns else 0.0)
        detail = f"Type overlap: {shared_types or 'none'}; entity overlap: {len(all_shared)} shared."

    # Detect hard type mismatches -- these are the key false-positive killers.
    # A mismatch is when the article is strongly one type and the market is
    # strongly a *different* type with zero overlap in entities of those types.
    type_mismatch = False
    if a_types and m_types and not shared_types:
        # Both have strong types but zero overlap
        # Check specifically for the worst offenders: sports vs non-sports
        if ENTITY_TYPE_SPORTS in a_types and ENTITY_TYPE_SPORTS not in m_types:
            type_mismatch = True
            detail = f"Article is sports ({article_entities['sports_entities']}), market is {m_types}."
        elif ENTITY_TYPE_SPORTS in m_types and ENTITY_TYPE_SPORTS not in a_types:
            type_mismatch = True
            detail = f"Market is sports ({market_entities['sports_entities']}), article is {a_types}."
        elif ENTITY_TYPE_POLITICS in a_types and ENTITY_TYPE_POLITICS not in m_types and not shared_nouns:
            type_mismatch = True
            detail = f"Article is political, market is {m_types}; no shared entities."
        elif ENTITY_TYPE_CRYPTO in a_types and ENTITY_TYPE_CRYPTO not in m_types and not shared_nouns:
            type_mismatch = True
            detail = f"Article is crypto, market is {m_types}; no shared entities."
        # Generic cross-type mismatch when zero entity overlap
        if not type_mismatch and not all_shared:
            type_mismatch = True
            detail = f"Article types {a_types} vs market types {m_types}; no shared entities."

    # Intra-sports league mismatch: both are "sports" but different leagues
    # with no shared team/entity overlap (e.g. Padres/MLB vs Spurs/NBA).
    # This is the specific scenario causing the Padres-vs-Spurs false positives.
    if (
        not type_mismatch
        and ENTITY_TYPE_SPORTS in a_types
        and ENTITY_TYPE_SPORTS in m_types
    ):
        a_leagues = article_entities.get("sports_leagues", set())
        m_leagues = market_entities.get("sports_leagues", set())
        shared_sports = article_entities.get("sports_entities", set()) & market_entities.get("sports_entities", set())
        if a_leagues and m_leagues and not (a_leagues & m_leagues) and not shared_sports:
            type_mismatch = True
            detail = (
                f"Intra-sports league mismatch: article leagues {a_leagues} "
                f"vs market leagues {m_leagues}; no shared sports entities."
            )

    if type_mismatch:
        score = max(0.0, score * 0.3)  # heavy penalty

    return {
        "entity_overlap_score": round(min(1.0, max(0.0, score)), 4),
        "type_match": bool(shared_types),
        "type_mismatch": type_mismatch,
        "shared_entities": sorted(all_shared),
        "shared_types": shared_types,
        "article_types": a_types,
        "market_types": m_types,
        "detail": detail,
    }


# ---------------------------------------------------------------------------
# Market-type detection (keyword-based domain classifier)
# ---------------------------------------------------------------------------
# Additional keyword sets for the market-type classifier.  These are broader
# than the entity-overlap sets above because we only need to detect the
# *domain* (not specific entity matches).

_MT_SPORTS_KEYWORDS: frozenset[str] = frozenset({
    "win", "championship", "game", "match", "playoff", "playoffs",
    "tournament", "mvp", "score", "season", "league", "finals",
    "super bowl", "world series", "world cup",
    "football", "basketball", "baseball", "hockey", "soccer", "tennis",
    "golf", "mma", "ufc", "boxing", "cricket", "rugby", "nascar",
    "formula 1", "f1",
    "nfl", "nba", "mlb", "nhl", "ncaa", "mls", "epl", "premier league",
    "la liga", "serie a", "bundesliga", "champions league", "uefa",
    "injury", "roster", "draft", "trade deadline", "free agent",
    "head coach", "starting lineup",
})

_MT_POLITICS_KEYWORDS: frozenset[str] = frozenset({
    "election", "president", "presidential", "vote", "voter", "voting",
    "congress", "senate", "senator", "representative", "governor",
    "policy", "legislation", "bill", "law", "executive order",
    "cabinet", "supreme court", "parliament", "prime minister",
    "democrat", "republican", "gop", "dnc", "rnc",
    "war", "treaty", "ceasefire", "sanctions", "nato", "united nations",
    "geopolitical", "diplomacy", "diplomatic", "ambassador",
    "ukraine", "russia", "china", "taiwan", "iran", "israel",
    "palestine", "north korea", "eu", "european union",
    "tariff", "trade war", "immigration", "border",
    "poll", "polling", "approval rating", "impeach",
})

_MT_CRYPTO_KEYWORDS: frozenset[str] = frozenset({
    "bitcoin", "btc", "ethereum", "eth", "solana", "sol", "xrp",
    "crypto", "cryptocurrency", "blockchain", "defi", "nft",
    "token", "altcoin", "stablecoin", "usdt", "usdc",
    "binance", "coinbase", "kraken",
    "market cap", "halving", "mining",
    "fed", "federal reserve", "interest rate", "rate cut", "rate hike",
    "inflation", "cpi", "gdp", "treasury", "yield curve",
    "sec", "etf", "spot etf",
    "price above", "price below", "price of",
})

_MT_ENTERTAINMENT_KEYWORDS: frozenset[str] = frozenset({
    "oscar", "oscars", "academy award", "grammy", "grammys",
    "emmy", "emmys", "golden globe", "bafta",
    "box office", "movie", "film", "album", "song", "artist",
    "streaming", "netflix", "disney", "spotify",
    "billboard", "chart", "rating", "ratings",
    "celebrity", "actor", "actress", "director",
    "tv show", "series premiere", "season finale",
    "award show", "nominee", "nomination",
})


def detect_market_type(
    question: str = "",
    slug: str = "",
    event_title: str = "",
    category: str = "",
) -> str:
    """Detect the market domain from its textual metadata.

    Returns one of: ``"sports"``, ``"politics"``, ``"crypto"``,
    ``"entertainment"``, ``"general"``.

    Uses simple keyword matching against the combined text.  The first
    domain with >= 2 keyword hits wins; ties are broken by hit count
    (highest wins).  If nothing reaches the threshold the Polymarket
    ``category`` field is used as a fallback.
    """
    text = " ".join([
        question.lower(),
        slug.lower().replace("-", " "),
        event_title.lower(),
        category.lower(),
    ])

    # Count hits per domain.  We require >=2 keyword matches to reduce
    # false positives (a single word like "game" could appear anywhere).
    # Uses _keyword_in_text for word-boundary-aware matching on short tokens.
    domain_hits: list[tuple[str, int]] = []
    for domain, keywords in [
        ("sports", _MT_SPORTS_KEYWORDS),
        ("politics", _MT_POLITICS_KEYWORDS),
        ("crypto", _MT_CRYPTO_KEYWORDS),
        ("entertainment", _MT_ENTERTAINMENT_KEYWORDS),
    ]:
        hits = sum(1 for kw in keywords if _keyword_in_text(kw, text))
        if hits >= 2:
            domain_hits.append((domain, hits))

    if not domain_hits:
        # Fallback: check the Polymarket category field directly.
        cat_lower = category.strip().lower()
        cat_map = {
            "sports": "sports",
            "politics": "politics",
            "pop culture": "entertainment",
            "entertainment": "entertainment",
            "crypto": "crypto",
            "business": "crypto",  # finance/business maps to crypto prompt
        }
        return cat_map.get(cat_lower, "general")

    # Return the domain with the most keyword hits.
    domain_hits.sort(key=lambda x: x[1], reverse=True)
    return domain_hits[0][0]


# ---------------------------------------------------------------------------
# Domain-specific LLM system prompts for reranking
# ---------------------------------------------------------------------------

_BASE_SCORING_RULES = (
    "SCORING SCALE:\n"
    "- 1.0 = article is DIRECTLY about the specific event/question the market asks.\n"
    "- 0.7-0.9 = article provides strong, actionable information for this market.\n"
    "- 0.4-0.6 = article is tangentially related (same broad topic, different specifics).\n"
    "- 0.1-0.3 = weak connection; same domain but different event/entity.\n"
    "- 0.0 = completely unrelated.\n\n"
    "IMPORTANT:\n"
    "- Be strict: most articles are NOT directly relevant to most markets.\n"
    "- Only score high when the article and market reference the same "
    "underlying event, not just the same broad category.\n"
)

_DOMAIN_SYSTEM_PROMPTS: Dict[str, str] = {
    "sports": (
        "You are a sports prediction market relevance scorer. "
        "Given a news article and a list of prediction market questions about sports, "
        "score how relevant the article is to each market.\n\n"
        + _BASE_SCORING_RULES
        + "SPORTS-SPECIFIC SIGNALS TO PRIORITIZE:\n"
        "- Injury reports and player availability (high impact on game outcomes)\n"
        "- Lineup changes, rotation decisions, and coaching adjustments\n"
        "- Recent team/player performance trends and streaks\n"
        "- Head-to-head records between specific teams or players\n"
        "- Weather/venue conditions that affect gameplay\n"
        "- Suspension, ejection, or disciplinary news\n"
        "- Trade deadline moves or roster transactions\n\n"
        "SPORTS PITFALLS:\n"
        "- Do NOT conflate different sports (e.g., NFL vs NBA).\n"
        "- Do NOT treat general league news as relevant to a specific game market.\n"
        "- Verify the article mentions the EXACT teams/players the market asks about."
    ),
    "politics": (
        "You are a political prediction market relevance scorer. "
        "Given a news article and a list of prediction market questions about "
        "politics or geopolitics, score how relevant the article is to each market.\n\n"
        + _BASE_SCORING_RULES
        + "POLITICS-SPECIFIC SIGNALS TO PRIORITIZE:\n"
        "- Polling data, approval ratings, and electoral forecasts\n"
        "- Policy announcements, executive orders, and legislative votes\n"
        "- Endorsements, campaign developments, and fundraising numbers\n"
        "- Legislative actions: bill introductions, committee votes, floor votes\n"
        "- Diplomatic events: treaties, summits, sanctions, military actions\n"
        "- Personnel changes: cabinet appointments, resignations, firings\n\n"
        "POLITICS PITFALLS:\n"
        "- Do NOT treat general political commentary as evidence for a specific outcome.\n"
        "- Polls from one race are NOT relevant to a different race or office.\n"
        "- Opinion pieces and editorials carry less weight than factual reporting.\n"
        "- Verify the article discusses the EXACT election, candidate, or policy "
        "the market asks about."
    ),
    "crypto": (
        "You are a crypto/finance prediction market relevance scorer. "
        "Given a news article and a list of prediction market questions about "
        "cryptocurrency or financial markets, score how relevant the article is "
        "to each market.\n\n"
        + _BASE_SCORING_RULES
        + "CRYPTO/FINANCE-SPECIFIC SIGNALS TO PRIORITIZE:\n"
        "- Regulatory news: SEC actions, ETF approvals/rejections, legislation\n"
        "- Adoption metrics: institutional purchases, corporate treasury moves\n"
        "- Technical developments: protocol upgrades, network milestones, forks\n"
        "- Macro indicators: Fed rate decisions, CPI data, employment reports\n"
        "- Exchange-specific news: listings, delistings, hacks, volume spikes\n"
        "- On-chain data signals: whale movements, supply dynamics\n\n"
        "CRYPTO/FINANCE PITFALLS:\n"
        "- Price prediction articles without new information are low-relevance.\n"
        "- News about one token is NOT automatically relevant to another token's market.\n"
        "- General macro news is only relevant if the market specifically asks about "
        "macro-sensitive outcomes (e.g., 'Will BTC exceed $X by date Y?').\n"
        "- Verify the article discusses the EXACT asset, metric, or threshold "
        "the market asks about."
    ),
    "entertainment": (
        "You are an entertainment prediction market relevance scorer. "
        "Given a news article and a list of prediction market questions about "
        "entertainment, awards, or pop culture, score how relevant the article "
        "is to each market.\n\n"
        + _BASE_SCORING_RULES
        + "ENTERTAINMENT-SPECIFIC SIGNALS TO PRIORITIZE:\n"
        "- Critic reviews, aggregated scores (Rotten Tomatoes, Metacritic)\n"
        "- Industry trends: box office tracking, streaming viewership data\n"
        "- Historical award patterns: precursor awards, nomination correlations\n"
        "- Insider reports: guild voting trends, campaign buzz, screening reactions\n"
        "- Official announcements: nominations, shortlists, eligibility decisions\n\n"
        "ENTERTAINMENT PITFALLS:\n"
        "- Celebrity gossip is NOT relevant to box office or award markets.\n"
        "- Reviews of one movie are NOT relevant to a different movie's market.\n"
        "- Verify the article discusses the EXACT title, artist, or award "
        "the market asks about."
    ),
    "general": (
        "You are a prediction market relevance scorer. "
        "Given a news article and a list of prediction market questions, "
        "score how relevant the article is to each market.\n\n"
        + _BASE_SCORING_RULES
        + "GENERAL GUIDANCE:\n"
        "- Focus on whether the article provides new, factual information that "
        "would shift a reasonable person's probability estimate for the market.\n"
        "- Consider timeliness: breaking news is more relevant than background context.\n"
        "- Look for causal connections, not just topical overlap."
    ),
}


@dataclass
class RerankedCandidate:
    """A candidate after LLM reranking."""

    candidate: RetrievalCandidate
    relevance: float  # 0-1 from LLM
    rationale: str
    rerank_score: float  # Final combined score
    used_llm: bool = False
    entity_overlap: Optional[Dict[str, Any]] = field(default_factory=lambda: None)

    @property
    def market_id(self) -> str:
        return self.candidate.market_id


# LLM reranking schema
RERANK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "pairs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {
                        "type": "integer",
                        "description": "The index of this market pair (0-based).",
                    },
                    "relevance": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": (
                            "How relevant this news article is to this specific market. "
                            "1.0 = directly about the same event/question. "
                            "0.0 = completely unrelated."
                        ),
                    },
                    "rationale": {
                        "type": "string",
                        "description": "One sentence explaining why this is or isn't relevant.",
                    },
                },
                "required": ["index", "relevance", "rationale"],
            },
        },
    },
    "required": ["pairs"],
}


class Reranker:
    """LLM-based reranking of retrieval candidates with entity pre-filtering."""

    def __init__(self) -> None:
        # Set after each rerank() call so callers (e.g. workflow orchestrator)
        # can read the detected domain for evidence/logging.
        self._last_detected_market_type: str = "general"

    @property
    def last_detected_market_type(self) -> str:
        """The market type detected during the most recent rerank() call."""
        return self._last_detected_market_type

    async def rerank(
        self,
        article_title: str,
        article_summary: str,
        candidates: list[RetrievalCandidate],
        top_n: int = 5,
        model: Optional[str] = None,
        allow_llm: bool = True,
    ) -> list[RerankedCandidate]:
        """Rerank candidates using entity overlap pre-filter then LLM.

        Entity-type overlap is computed BEFORE LLM calls to reject clearly
        mismatched pairs (e.g. baseball article matched to basketball market)
        and save LLM budget.

        Falls back to retrieval scores if LLM is unavailable.

        Args:
            article_title: News article title.
            article_summary: Article summary text.
            candidates: Candidates from hybrid retriever.
            top_n: Number of top candidates to return.
            model: LLM model override.

        Returns:
            Top-N RerankedCandidate sorted by rerank_score desc.
        """
        if not candidates:
            return []

        # ---- Phase 1: Entity-type overlap pre-filter (no LLM cost) ----
        article_text = f"{article_title} {article_summary or ''}"
        article_entities = _extract_entities(article_text)

        filtered_candidates: list[RetrievalCandidate] = []
        entity_overlaps: Dict[str, Dict[str, Any]] = {}  # market_id -> overlap data
        skipped_by_entity: int = 0

        for c in candidates:
            market_text = f"{c.question} {c.event_title or ''} {c.category or ''} {' '.join(c.tags or [])}"
            market_entities = _extract_entities(market_text)
            overlap = _compute_entity_overlap(article_entities, market_entities)

            entity_overlaps[c.market_id] = overlap

            # Skip pairs with hard type mismatch AND very low overlap score
            if overlap["type_mismatch"] and overlap["entity_overlap_score"] < ENTITY_OVERLAP_MIN_SCORE:
                skipped_by_entity += 1
                logger.debug(
                    "Entity pre-filter skipped market %s: %s (score=%.3f)",
                    c.market_id,
                    overlap["detail"],
                    overlap["entity_overlap_score"],
                )
                continue

            filtered_candidates.append(c)

        if skipped_by_entity:
            logger.info(
                "Entity pre-filter: skipped %d/%d candidates for article '%s'",
                skipped_by_entity,
                len(candidates),
                article_title[:80],
            )

        if not filtered_candidates:
            # All candidates filtered out -- return empty scored results so
            # callers can still see what was rejected in debug views.
            return []

        # ---- Phase 1b: Detect market type for prompt selection ----
        # Done here so the result is available whether LLM succeeds or not.
        if filtered_candidates:
            combined_questions = " ".join(c.question for c in filtered_candidates[:5])
            combined_slugs = " ".join(getattr(c, "slug", "") or "" for c in filtered_candidates[:5])
            combined_events = " ".join(c.event_title or "" for c in filtered_candidates[:5])
            combined_categories = " ".join(c.category or "" for c in filtered_candidates[:5])
            self._last_detected_market_type = detect_market_type(
                question=combined_questions,
                slug=combined_slugs,
                event_title=combined_events,
                category=combined_categories,
            )

        # ---- Phase 2: LLM reranking on surviving candidates ----
        reranked = None
        if allow_llm:
            reranked = await self._rerank_llm(
                article_title, article_summary, filtered_candidates, model=model
            )

        if reranked is not None:
            # Attach entity overlap data to each result for debugging
            for rc in reranked:
                overlap = entity_overlaps.get(rc.candidate.market_id)
                if overlap is not None:
                    rc.entity_overlap = overlap
            reranked.sort(key=lambda r: r.rerank_score, reverse=True)
            return reranked[:top_n]

        # Fallback: use retrieval scores directly, still incorporating entity overlap
        results = []
        for c in filtered_candidates:
            overlap = entity_overlaps.get(c.market_id, {})
            entity_score = overlap.get("entity_overlap_score", 0.5)
            # Blend retrieval score with entity overlap (20% weight for entity)
            blended_score = 0.80 * c.combined_score + 0.20 * entity_score
            results.append(
                RerankedCandidate(
                    candidate=c,
                    relevance=c.combined_score,
                    rationale="Retrieval score + entity overlap (LLM unavailable)",
                    rerank_score=blended_score,
                    used_llm=False,
                    entity_overlap=overlap if overlap else None,
                )
            )
        results.sort(key=lambda r: r.rerank_score, reverse=True)
        return results[:top_n]

    async def _rerank_llm(
        self,
        article_title: str,
        article_summary: str,
        candidates: list[RetrievalCandidate],
        model: Optional[str] = None,
    ) -> Optional[list[RerankedCandidate]]:
        """LLM-based reranking with market-type-aware prompting."""
        try:
            from services.ai import get_llm_manager
            from services.ai.llm_provider import LLMMessage

            manager = get_llm_manager()
            if not manager.is_available():
                return None
        except Exception:
            return None

        # Detect the dominant market type from the candidate set.
        # Combine text from the first few candidates for a robust signal.
        market_type = "general"
        if candidates:
            combined_questions = " ".join(c.question for c in candidates[:5])
            combined_slugs = " ".join(getattr(c, "slug", "") or "" for c in candidates[:5])
            combined_events = " ".join(c.event_title or "" for c in candidates[:5])
            combined_categories = " ".join(c.category or "" for c in candidates[:5])
            market_type = detect_market_type(
                question=combined_questions,
                slug=combined_slugs,
                event_title=combined_events,
                category=combined_categories,
            )

        system_prompt = _DOMAIN_SYSTEM_PROMPTS.get(market_type, _DOMAIN_SYSTEM_PROMPTS["general"])

        logger.debug(
            "Reranker detected market_type=%s for article=%r (%d candidates)",
            market_type,
            article_title[:80],
            len(candidates),
        )

        # Store detected market type so callers can propagate it into evidence.
        self._last_detected_market_type = market_type

        # Build market list
        market_lines = []
        for i, c in enumerate(candidates):
            market_lines.append(
                f"[{i}] {c.question}"
                + (f" (Event: {c.event_title})" if c.event_title else "")
                + (f" [{c.category}]" if c.category else "")
            )

        user_prompt = f"NEWS ARTICLE:\n  Title: {article_title}\n"
        if article_summary:
            user_prompt += f"  Summary: {article_summary[:400]}\n"

        user_prompt += (
            f"\nMARKET QUESTIONS ({len(candidates)}):\n"
            + "\n".join(market_lines)
            + "\n\nScore each market's relevance to this article."
        )

        try:
            result = await asyncio.wait_for(
                manager.structured_output(
                    messages=[
                        LLMMessage(role="system", content=system_prompt),
                        LLMMessage(role="user", content=user_prompt),
                    ],
                    schema=RERANK_SCHEMA,
                    model=model,
                    purpose="news_workflow_rerank",
                ),
                timeout=_LLM_CALL_TIMEOUT_SECONDS,
            )

            pairs = result.get("pairs", [])
            reranked: list[RerankedCandidate] = []

            for pair in pairs:
                idx = pair.get("index", -1)
                if idx < 0 or idx >= len(candidates):
                    continue
                relevance = float(pair.get("relevance", 0.0))
                rationale = pair.get("rationale", "")

                c = candidates[idx]
                # Combine retrieval score with LLM relevance
                # Entity overlap penalty is applied post-hoc if type_mismatch
                rerank_score = 0.3 * c.combined_score + 0.7 * relevance

                reranked.append(
                    RerankedCandidate(
                        candidate=c,
                        relevance=relevance,
                        rationale=rationale,
                        rerank_score=rerank_score,
                        used_llm=True,
                    )
                )

            # Include candidates not scored by LLM (fallback)
            scored_indices = {p.get("index") for p in pairs}
            for i, c in enumerate(candidates):
                if i not in scored_indices:
                    reranked.append(
                        RerankedCandidate(
                            candidate=c,
                            relevance=0.0,
                            rationale="Not scored by LLM",
                            rerank_score=c.combined_score * 0.3,
                            used_llm=False,
                        )
                    )

            return reranked

        except asyncio.TimeoutError:
            logger.debug(
                "LLM reranking timed out after %.1fs",
                _LLM_CALL_TIMEOUT_SECONDS,
            )
            return None
        except Exception as e:
            logger.debug("LLM reranking failed: %s", e)
            return None


# Singleton
reranker = Reranker()
