import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from utils.market_urls import (  # noqa: E402
    attach_market_links_to_opportunity_dict,
    build_kalshi_market_url,
    build_polymarket_market_url,
)


def test_build_polymarket_market_url_prefers_canonical_paths():
    assert (
        build_polymarket_market_url(
            event_slug="how-many-people-will-trump-deport-in-2025",
            market_slug="will-trump-deport-less-than-250000",
        )
        == "https://polymarket.com/event/how-many-people-will-trump-deport-in-2025/will-trump-deport-less-than-250000"
    )
    assert (
        build_polymarket_market_url(market_slug="will-trump-deport-less-than-250000")
        == "https://polymarket.com/market/will-trump-deport-less-than-250000"
    )
    assert (
        build_polymarket_market_url(event_slug="how-many-people-will-trump-deport-in-2025")
        == "https://polymarket.com/event/how-many-people-will-trump-deport-in-2025"
    )


def test_build_polymarket_market_url_rejects_non_slug_ids():
    assert (
        build_polymarket_market_url(
            condition_id="0x064d33e3f5703792aafa92bfb0ee10e08f461b1b34c02c1f02671892ede1609a"
        )
        is None
    )
    assert build_polymarket_market_url(market_id="517310") is None
    assert (
        build_polymarket_market_url(
            market_id="will-joe-biden-get-coronavirus-before-the-election"
        )
        == "https://polymarket.com/market/will-joe-biden-get-coronavirus-before-the-election"
    )


def test_build_kalshi_market_url_uses_ticker_only_route():
    assert (
        build_kalshi_market_url(market_ticker="KXELONMARS-99")
        == "https://kalshi.com/markets/KXELONMARS-99"
    )
    assert (
        build_kalshi_market_url(market_ticker="KXELONMARS-99_yes")
        == "https://kalshi.com/markets/KXELONMARS-99"
    )
    assert (
        build_kalshi_market_url(event_ticker="KXELONMARS-99")
        == "https://kalshi.com/markets/KXELONMARS-99"
    )
    assert build_kalshi_market_url(market_ticker="kasimpasa vs karagumruk winner?") is None


def test_attach_market_links_keeps_api_url_and_fills_platform_links():
    opportunity = {
        "event_slug": "how-many-people-will-trump-deport-in-2025",
        "markets": [
            {
                "platform": "polymarket",
                "slug": "will-trump-deport-less-than-250000",
                "url": "https://polymarket.com/market/will-trump-deport-less-than-250000",
            },
            {
                "platform": "kalshi",
                "id": "KXELONMARS-99_yes",
            },
        ],
    }

    enriched = attach_market_links_to_opportunity_dict(opportunity)
    assert (
        enriched["polymarket_url"]
        == "https://polymarket.com/market/will-trump-deport-less-than-250000"
    )
    assert enriched["kalshi_url"] == "https://kalshi.com/markets/KXELONMARS-99"
    assert enriched["markets"][0]["url"] == enriched["polymarket_url"]
    assert enriched["markets"][1]["url"] == enriched["kalshi_url"]
