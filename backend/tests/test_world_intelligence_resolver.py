import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.world_intelligence import resolver


def _opportunity_with_market(
    market_id: str = "cond_1",
    yes_token: str = "tok_yes",
    no_token: str = "tok_no",
    yes_price: float = 0.42,
    no_price: float = 0.58,
):
    return SimpleNamespace(
        event_id="evt_1",
        event_slug="event-1",
        event_title="Event One",
        category="politics",
        title="Will event happen?",
        min_liquidity=15000.0,
        markets=[
            {
                "id": market_id,
                "question": "Will event happen?",
                "liquidity": 15000.0,
            }
        ],
        positions_to_take=[
            {
                "market_id": market_id,
                "market_question": "Will event happen?",
                "outcome": "YES",
                "token_id": yes_token,
                "price": yes_price,
            },
            {
                "market_id": market_id,
                "market_question": "Will event happen?",
                "outcome": "NO",
                "token_id": no_token,
                "price": no_price,
            },
        ],
    )


@pytest.mark.asyncio
async def test_resolve_world_signal_opportunities_tradable(monkeypatch):
    async def _fake_read_scanner_snapshot(_session):
        return [_opportunity_with_market()], {"status": "ok"}

    monkeypatch.setattr(resolver, "read_scanner_snapshot", _fake_read_scanner_snapshot)

    rows = await resolver.resolve_world_signal_opportunities(
        session=object(),  # only passed through to mocked read_scanner_snapshot
        signals=[
            {
                "signal_id": "wi_1",
                "signal_type": "conflict",
                "severity": 0.9,
                "country": "USA",
                "source": "acled",
                "title": "Conflict escalation",
                "description": "Escalation detected",
                "related_market_ids": ["cond_1"],
                "market_relevance_score": 0.8,
                "metadata": {},
            }
        ],
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["tradable"] is True
    assert row["direction"] == "buy_yes"
    assert row["token_id"] == "tok_yes"
    assert row["entry_price"] == pytest.approx(0.42)
    assert row["resolver_status"] == "tradable"


@pytest.mark.asyncio
async def test_resolve_world_signal_opportunities_marks_missing_fields(monkeypatch):
    # No token IDs available -> unresolved candidate.
    opp = _opportunity_with_market(yes_token="", no_token="")

    async def _fake_read_scanner_snapshot(_session):
        return [opp], {"status": "ok"}

    monkeypatch.setattr(resolver, "read_scanner_snapshot", _fake_read_scanner_snapshot)

    rows = await resolver.resolve_world_signal_opportunities(
        session=object(),
        signals=[
            {
                "signal_id": "wi_2",
                "signal_type": "tension",
                "severity": 0.75,
                "country": "UKR-RUS",
                "source": "gdelt",
                "title": "Tension spike",
                "description": "Tension increased",
                "related_market_ids": ["cond_1"],
                "market_relevance_score": 0.7,
                "metadata": {},
            }
        ],
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["tradable"] is False
    assert "token_id" in row["missing_fields"]


def test_infer_direction_from_anomaly_zscore():
    signal = {
        "signal_type": "anomaly",
        "severity": 0.6,
        "metadata": {"z_score": -3.0},
    }
    assert resolver.infer_direction(signal) == "buy_no"

