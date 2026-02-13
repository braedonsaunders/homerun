import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from api.routes import _derive_opportunity_sub_strategy  # noqa: E402


def test_derive_temporal_decay_subfilters():
    certainty = {
        "strategy": "temporal_decay",
        "title": "Certainty Shock: Government shutdown?",
        "positions_to_take": [],
    }
    decay = {
        "strategy": "temporal_decay",
        "title": "Temporal Decay: BTC above 120k by March?",
        "positions_to_take": [],
    }

    assert _derive_opportunity_sub_strategy(certainty) == "certainty_shock"
    assert _derive_opportunity_sub_strategy(decay) == "decay_curve"


def test_derive_highfreq_substrategy_from_metadata():
    opp = {
        "strategy": "btc_eth_highfreq",
        "title": "BTC/ETH HF Directional",
        "positions_to_take": [
            {"action": "BUY", "outcome": "YES"},
            {
                "_highfreq_metadata": True,
                "sub_strategy": "directional_edge",
            },
        ],
    }

    assert _derive_opportunity_sub_strategy(opp) == "directional_edge"


def test_derive_news_edge_direction():
    opp = {
        "strategy": "news_edge",
        "title": "News Edge: Example",
        "positions_to_take": [
            {
                "action": "BUY",
                "outcome": "YES",
                "_news_edge": {"direction": "buy_yes"},
            }
        ],
    }

    assert _derive_opportunity_sub_strategy(opp) == "buy_yes"


def test_derive_cross_platform_leg_shape():
    opp = {
        "strategy": "cross_platform",
        "title": "Cross-Platform: Example",
        "positions_to_take": [
            {"platform": "polymarket", "outcome": "YES"},
            {"platform": "kalshi", "outcome": "NO"},
        ],
    }

    assert _derive_opportunity_sub_strategy(opp) == "poly_yes_kalshi_no"
