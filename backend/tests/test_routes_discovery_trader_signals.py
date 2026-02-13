import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from api import routes_discovery


class _RowsResult:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


@pytest.mark.asyncio
async def test_tracked_trader_opportunities_include_source_and_validation_metadata(
    monkeypatch,
):
    opportunities = [
        {
            "id": "sig-good",
            "market_id": "0xmarket",
            "market_question": "Will BTC be above $120k?",
            "wallets": ["0xpool", "0xtracked"],
            "signal_type": "multi_wallet_buy",
            "outcome": "YES",
            "yes_price": 0.62,
            "no_price": 0.38,
        },
        {
            "id": "sig-bad",
            "market_id": "",
            "market_question": "",
            "wallets": ["0xunknown"],
            "signal_type": "",
            "outcome": None,
            "yes_price": 1.2,
            "no_price": -0.1,
        },
    ]

    session = SimpleNamespace(
        execute=AsyncMock(
            side_effect=[
                _RowsResult([("0xpool", True), ("0xtracked", False)]),
                _RowsResult([("0xtracked",)]),
                _RowsResult([("0xtracked", "group-1")]),
            ]
        )
    )

    monkeypatch.setattr(
        routes_discovery.smart_wallet_pool,
        "get_tracked_trader_opportunities",
        AsyncMock(return_value=opportunities),
    )
    monkeypatch.setattr(
        routes_discovery,
        "_load_scanner_market_history",
        AsyncMock(return_value={}),
    )
    monkeypatch.setattr(
        routes_discovery,
        "_attach_signal_market_metadata",
        AsyncMock(side_effect=lambda rows: rows),
    )
    monkeypatch.setattr(
        routes_discovery,
        "_attach_activity_history_fallback",
        AsyncMock(return_value=None),
    )

    payload = await routes_discovery.get_tracked_trader_opportunities(
        limit=50,
        min_tier="WATCH",
        session=session,
    )

    assert payload["total"] == 2
    by_id = {row["id"]: row for row in payload["opportunities"]}

    good = by_id["sig-good"]
    assert good["source_flags"]["from_pool"] is True
    assert good["source_flags"]["from_tracked_traders"] is True
    assert good["source_flags"]["qualified"] is True
    assert good["is_valid"] is True
    assert good["is_tradeable"] is True
    assert good["source_breakdown"]["pool_wallets"] == 1
    assert good["source_breakdown"]["tracked_wallets"] == 1

    bad = by_id["sig-bad"]
    assert bad["source_flags"]["qualified"] is False
    assert bad["is_valid"] is False
    assert bad["is_tradeable"] is False
    assert "missing_market_id" in bad["validation_reasons"]
    assert "missing_direction" in bad["validation_reasons"]
    assert "price_out_of_bounds" in bad["validation_reasons"]
    assert "unqualified_wallet_source" in bad["validation_reasons"]


@pytest.mark.asyncio
async def test_insider_opportunities_source_metadata_uses_wallet_payloads(monkeypatch):
    payload = {
        "total": 1,
        "offset": 0,
        "limit": 50,
        "opportunities": [
            {
                "id": "ins-1",
                "market_id": "0xinsider",
                "market_question": "Will ETH close above $5k this week?",
                "direction": "buy_yes",
                "confidence": 0.78,
                "entry_price": 0.24,
                "wallet_addresses": ["0xalpha", "0xbeta"],
                "wallets": [
                    {"address": "0xalpha", "username": "alpha"},
                ],
                "top_wallet": {"address": "0xbeta", "username": "beta"},
            }
        ],
    }

    session = SimpleNamespace(
        execute=AsyncMock(
            side_effect=[
                _RowsResult([("0xalpha", False), ("0xbeta", True)]),
                _RowsResult([("0xalpha",)]),
                _RowsResult([("0xbeta", "group-2")]),
            ]
        )
    )

    monkeypatch.setattr(
        routes_discovery.insider_detector,
        "list_opportunities",
        AsyncMock(return_value=payload),
    )
    monkeypatch.setattr(
        routes_discovery,
        "_load_scanner_market_history",
        AsyncMock(return_value={}),
    )
    monkeypatch.setattr(
        routes_discovery,
        "_attach_signal_market_metadata",
        AsyncMock(side_effect=lambda rows: rows),
    )
    monkeypatch.setattr(
        routes_discovery,
        "_attach_activity_history_fallback",
        AsyncMock(return_value=None),
    )

    result = await routes_discovery.get_insider_opportunities(
        limit=50,
        offset=0,
        min_confidence=0.0,
        direction=None,
        max_age_minutes=180,
        session=session,
    )

    opportunity = result["opportunities"][0]
    assert opportunity["source_breakdown"]["wallets_considered"] == 2
    assert opportunity["source_breakdown"]["pool_wallets"] == 1
    assert opportunity["source_breakdown"]["tracked_wallets"] == 1
    assert opportunity["source_breakdown"]["group_wallets"] == 1
    assert opportunity["source_flags"]["from_trader_groups"] is True
    assert opportunity["source_flags"]["qualified"] is True
    assert opportunity["is_valid"] is True
    assert opportunity["is_tradeable"] is True


@pytest.mark.asyncio
async def test_attach_signal_market_metadata_uses_api_labels_and_backfill_prices(monkeypatch):
    rows = [
        {
            "id": "sig-1",
            "market_id": "0xabc",
            "yes_price": 0.61,
            "no_price": 0.39,
        },
        {
            "id": "sig-2",
            "market_id": "123",
        },
    ]

    monkeypatch.setattr(
        routes_discovery.polymarket_client,
        "get_market_by_condition_id",
        AsyncMock(
            return_value={
                "outcomes": ["Team A", "Team B"],
                "outcome_prices": [0.52, 0.48],
            }
        ),
    )
    monkeypatch.setattr(
        routes_discovery.polymarket_client,
        "get_market_by_token_id",
        AsyncMock(
            return_value={
                "outcomes": '["Over 2.5", "Under 2.5"]',
                "outcome_prices": "[0.44, 0.56]",
            }
        ),
    )

    result = await routes_discovery._attach_signal_market_metadata(rows)
    first, second = result

    assert first["yes_label"] == "Team A"
    assert first["no_label"] == "Team B"
    # Existing backfill prices stay authoritative.
    assert first["current_yes_price"] == pytest.approx(0.61)
    assert first["current_no_price"] == pytest.approx(0.39)

    assert second["yes_label"] == "Over 2.5"
    assert second["no_label"] == "Under 2.5"
    assert second["current_yes_price"] == pytest.approx(0.44)
    assert second["current_no_price"] == pytest.approx(0.56)
