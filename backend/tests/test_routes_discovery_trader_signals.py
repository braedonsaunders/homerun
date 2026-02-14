import sys
from datetime import datetime
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
async def test_tracked_trader_opportunities_include_filtered_passthrough(monkeypatch):
    opportunities = [
        {
            "id": "sig-filtered",
            "market_id": "0xfiltered",
            "market_question": "Will this be visible in filtered mode?",
            "wallets": ["0xpool"],
            "signal_type": "multi_wallet_buy",
            "outcome": "YES",
            "yes_price": 0.48,
            "no_price": 0.52,
            "is_active": False,
            "is_tradeable": False,
        }
    ]

    session = SimpleNamespace(
        execute=AsyncMock(
            side_effect=[
                _RowsResult([]),
                _RowsResult([]),
                _RowsResult([]),
            ]
        )
    )

    get_opps_mock = AsyncMock(return_value=opportunities)
    monkeypatch.setattr(
        routes_discovery.smart_wallet_pool,
        "get_tracked_trader_opportunities",
        get_opps_mock,
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
        limit=25,
        min_tier="WATCH",
        include_filtered=True,
        session=session,
    )

    get_opps_mock.assert_awaited_once_with(
        limit=25,
        min_tier="WATCH",
        include_filtered=True,
    )
    assert payload["total"] == 1
    assert payload["opportunities"][0]["id"] == "sig-filtered"


@pytest.mark.asyncio
async def test_tracked_trader_opportunities_source_filter_scopes_confluence(monkeypatch):
    opportunities = [
        {
            "id": "sig-tracked",
            "market_id": "0xtracked",
            "market_question": "Tracked signal",
            "wallets": ["0xtracked"],
            "signal_type": "multi_wallet_buy",
            "outcome": "YES",
            "yes_price": 0.51,
            "no_price": 0.49,
        },
        {
            "id": "sig-pool",
            "market_id": "0xpool",
            "market_question": "Pool signal",
            "wallets": ["0xpool"],
            "signal_type": "multi_wallet_buy",
            "outcome": "YES",
            "yes_price": 0.53,
            "no_price": 0.47,
        },
        {
            "id": "sig-unqualified",
            "market_id": "0xother",
            "market_question": "Unqualified signal",
            "wallets": ["0xother"],
            "signal_type": "multi_wallet_buy",
            "outcome": "YES",
            "yes_price": 0.55,
            "no_price": 0.45,
        },
    ]

    def make_session():
        return SimpleNamespace(
            execute=AsyncMock(
                side_effect=[
                    _RowsResult([("0xtracked", False), ("0xpool", True), ("0xother", False)]),
                    _RowsResult([("0xtracked",)]),
                    _RowsResult([]),
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

    payload_all = await routes_discovery.get_tracked_trader_opportunities(
        limit=50,
        min_tier="WATCH",
        source_filter="all",
        include_filtered=False,
        session=make_session(),
    )
    ids_all = {row["id"] for row in payload_all["opportunities"]}
    assert ids_all == {"sig-tracked", "sig-pool"}

    payload_tracked = await routes_discovery.get_tracked_trader_opportunities(
        limit=50,
        min_tier="WATCH",
        source_filter="tracked",
        include_filtered=False,
        session=make_session(),
    )
    assert {row["id"] for row in payload_tracked["opportunities"]} == {"sig-tracked"}

    payload_pool = await routes_discovery.get_tracked_trader_opportunities(
        limit=50,
        min_tier="WATCH",
        source_filter="pool",
        include_filtered=False,
        session=make_session(),
    )
    assert {row["id"] for row in payload_pool["opportunities"]} == {"sig-pool"}

    payload_legacy_confluence = await routes_discovery.get_tracked_trader_opportunities(
        limit=50,
        min_tier="WATCH",
        source_filter="confluence",
        include_filtered=False,
        session=make_session(),
    )
    assert {row["id"] for row in payload_legacy_confluence["opportunities"]} == {
        "sig-tracked",
        "sig-pool",
    }


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


@pytest.mark.asyncio
async def test_attach_activity_history_fallback_preserves_named_outcome_prices():
    rows = [
        {
            "id": "sig-tennis",
            "market_id": "0xabc",
            "outcome": "YES",
            "outcome_labels": ["Kecmanovic", "Shelton"],
            "yes_price": 0.27,
            "no_price": 0.73,
        }
    ]
    session = SimpleNamespace(
        execute=AsyncMock(
            return_value=_RowsResult(
                [
                    ("0xabc", datetime(2026, 2, 14, 2, 11, 9), "BUY", 0.27),
                    ("0xabc", datetime(2026, 2, 14, 2, 11, 49), "BUY", 0.73),
                ]
            )
        )
    )

    await routes_discovery._attach_activity_history_fallback(session, rows)

    row = rows[0]
    assert row["yes_price"] == pytest.approx(0.27)
    assert row["no_price"] == pytest.approx(0.73)
    assert row.get("price_history") is None


@pytest.mark.asyncio
async def test_attach_activity_history_fallback_populates_yes_no_when_missing():
    rows = [
        {
            "id": "sig-binary",
            "market_id": "0xdef",
            "outcome": "YES",
            "outcome_labels": ["Yes", "No"],
        }
    ]
    session = SimpleNamespace(
        execute=AsyncMock(
            return_value=_RowsResult(
                [
                    ("0xdef", datetime(2026, 2, 14, 2, 10, 0), "BUY", 0.61),
                    ("0xdef", datetime(2026, 2, 14, 2, 11, 0), "BUY", 0.62),
                ]
            )
        )
    )

    await routes_discovery._attach_activity_history_fallback(session, rows)

    row = rows[0]
    assert row["yes_price"] == pytest.approx(0.62)
    assert row["no_price"] == pytest.approx(0.38)
    assert row["current_yes_price"] == pytest.approx(0.62)
    assert row["current_no_price"] == pytest.approx(0.38)
    assert len(row.get("price_history") or []) == 2
