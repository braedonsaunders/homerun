from __future__ import annotations

import re
from collections.abc import Mapping, MutableMapping
from typing import Any
from urllib.parse import quote

POLYMARKET_BASE_URL = "https://polymarket.com"
KALSHI_BASE_URL = "https://kalshi.com"

_CONDITION_ID_RE = re.compile(r"^0x[0-9a-fA-F]+$")
_KALSHI_TICKER_RE = re.compile(r"^KX[A-Z0-9-]+$")
_POLYMARKET_SLUG_RE = re.compile(r"^(?=.*[a-z])[a-z0-9-]+$")


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _clean_segment(value: Any) -> str:
    return _clean_text(value).strip("/")


def _is_condition_id(value: str) -> bool:
    return bool(_CONDITION_ID_RE.fullmatch(value))


def _is_likely_kalshi_ticker(value: Any) -> bool:
    ticker = _clean_segment(value).upper()
    return bool(ticker and _KALSHI_TICKER_RE.fullmatch(ticker))


def _is_likely_polymarket_slug(value: Any) -> bool:
    slug = _clean_segment(value).lower()
    return bool(slug and _POLYMARKET_SLUG_RE.fullmatch(slug))


def _normalize_kalshi_ticker(value: Any) -> str:
    return re.sub(r"_(yes|no)$", "", _clean_segment(value), flags=re.IGNORECASE)


def derive_kalshi_event_ticker(market_ticker: Any) -> str:
    ticker = _normalize_kalshi_ticker(market_ticker)
    if not ticker:
        return ""

    # Kalshi event tickers are the first hyphen-separated segment of the
    # market ticker (e.g. "KXBTCD" from "KXBTCD-26FEB1314",
    # "KXATPCHALLENGERMATCH" from "KXATPCHALLENGERMATCH-26FEB14KASGOM-KAS").
    parts = [p for p in ticker.split("-") if p]
    if len(parts) <= 1:
        return ticker
    return parts[0]


def infer_market_platform(market: Mapping[str, Any]) -> str:
    explicit = _clean_segment(market.get("platform")).lower()
    if explicit == "kalshi":
        return "kalshi"
    if explicit == "polymarket":
        return "polymarket"

    condition_id = _clean_segment(market.get("condition_id") or market.get("conditionId"))
    if _is_condition_id(condition_id):
        return "polymarket"

    market_id = _clean_segment(market.get("id") or market.get("market_id") or market.get("ticker"))
    market_slug = _clean_segment(market.get("slug") or market.get("market_slug"))
    event_ticker = _clean_segment(market.get("event_ticker") or market.get("eventTicker"))
    if (
        _is_likely_kalshi_ticker(market_id)
        or _is_likely_kalshi_ticker(market_slug)
        or _is_likely_kalshi_ticker(event_ticker)
    ):
        return "kalshi"

    return "polymarket"


def build_polymarket_market_url(
    *,
    event_slug: Any = None,
    market_slug: Any = None,
    market_id: Any = None,
    condition_id: Any = None,
) -> str | None:
    event_slug_text = _clean_segment(event_slug).lower()
    market_slug_text = _clean_segment(market_slug).lower()
    market_id_text = _clean_segment(market_id).lower()
    condition_id_text = _clean_segment(condition_id).lower()

    if event_slug_text and market_slug_text and event_slug_text != market_slug_text:
        return f"{POLYMARKET_BASE_URL}/event/{quote(event_slug_text, safe='')}/{quote(market_slug_text, safe='')}"
    if market_slug_text:
        # /market/{market_slug} is stable and resolves to the canonical event path.
        return f"{POLYMARKET_BASE_URL}/market/{quote(market_slug_text, safe='')}"
    if event_slug_text:
        return f"{POLYMARKET_BASE_URL}/event/{quote(event_slug_text, safe='')}"

    # Never use numeric/condition IDs for website URLs; they 404 on polymarket.com.
    if not _is_condition_id(condition_id_text) and _is_likely_polymarket_slug(market_id_text):
        return f"{POLYMARKET_BASE_URL}/market/{quote(market_id_text, safe='')}"
    return None


def build_kalshi_market_url(
    *,
    market_ticker: Any = None,
    event_ticker: Any = None,
    event_slug: Any = None,
) -> str | None:
    # Kalshi website URLs resolve via the event ticker (lowercase), e.g.
    # https://kalshi.com/markets/kxbtcmaxy  — full market tickers 404.
    # Prefer the explicit event_ticker / event_slug from the API, then
    # fall back to deriving the event portion from the market ticker.
    event_ticker_text = (
        _clean_segment(event_ticker)
        or _clean_segment(event_slug)
        or derive_kalshi_event_ticker(_normalize_kalshi_ticker(market_ticker))
    )
    if _is_likely_kalshi_ticker(event_ticker_text):
        return f"{KALSHI_BASE_URL}/markets/{quote(event_ticker_text.lower(), safe='')}"

    return None


def build_market_url(
    market: Mapping[str, Any],
    *,
    opportunity_event_slug: Any = None,
) -> str | None:
    platform = infer_market_platform(market)
    if platform == "kalshi":
        # For Kalshi, "slug" / "market_slug" contain the full market ticker
        # (e.g. "KXPOLITICSMENTION-26FEB15-SHUT"), NOT the event slug.
        # Only pass explicitly event-scoped fields to avoid building URLs
        # from the full market ticker.
        return build_kalshi_market_url(
            market_ticker=market.get("id") or market.get("market_id") or market.get("ticker"),
            event_ticker=(
                market.get("event_ticker")
                or market.get("eventTicker")
            ),
            event_slug=(
                market.get("event_slug") or market.get("eventSlug")
            ),
        )

    return build_polymarket_market_url(
        event_slug=market.get("event_slug") or market.get("eventSlug") or opportunity_event_slug,
        market_slug=market.get("slug") or market.get("market_slug"),
        market_id=market.get("id") or market.get("market_id"),
        condition_id=market.get("condition_id") or market.get("conditionId"),
    )


def _clean_absolute_url(value: Any) -> str:
    text = _clean_text(value)
    if text.startswith("http://") or text.startswith("https://"):
        return text
    return ""


def attach_market_links_to_opportunity_dict(
    opportunity: MutableMapping[str, Any] | Mapping[str, Any],
) -> dict[str, Any]:
    data = dict(opportunity)
    event_slug = _clean_segment(data.get("event_slug") or data.get("eventSlug"))

    polymarket_url = _clean_absolute_url(data.get("polymarket_url"))
    kalshi_url = _clean_absolute_url(data.get("kalshi_url"))

    market_rows = data.get("markets")
    markets = market_rows if isinstance(market_rows, list) else []
    enriched_markets: list[Any] = []

    for market_entry in markets:
        if not isinstance(market_entry, Mapping):
            enriched_markets.append(market_entry)
            continue

        market_dict = dict(market_entry)
        platform = infer_market_platform(market_dict)
        market_dict["platform"] = platform

        market_url = _clean_absolute_url(market_dict.get("url") or market_dict.get("market_url"))
        if not market_url:
            market_url = build_market_url(market_dict, opportunity_event_slug=event_slug)

        if market_url:
            market_dict["url"] = market_url
            market_dict["market_url"] = market_url
            if platform == "polymarket" and not polymarket_url:
                polymarket_url = market_url
            if platform == "kalshi" and not kalshi_url:
                kalshi_url = market_url

        enriched_markets.append(market_dict)

    data["markets"] = enriched_markets
    data["polymarket_url"] = polymarket_url or None
    data["kalshi_url"] = kalshi_url or None
    return data


def serialize_opportunity_with_links(opportunity: Any) -> dict[str, Any]:
    if isinstance(opportunity, Mapping):
        payload = dict(opportunity)
    elif hasattr(opportunity, "model_dump"):
        payload = opportunity.model_dump(mode="json")
    else:
        raise TypeError("Unsupported opportunity payload type")
    return attach_market_links_to_opportunity_dict(payload)
