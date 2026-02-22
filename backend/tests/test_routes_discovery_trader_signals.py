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
async def test_load_tracked_trader_opportunities_delegates_to_strategy_pipeline(monkeypatch):
    rows = [{"id": "sig-1"}, {"id": "sig-2"}]
    loader = AsyncMock(return_value=rows)
    monkeypatch.setattr(
        routes_discovery.StrategySDK,
        "get_trader_strategy_signals",
        loader,
    )
    result = await routes_discovery._load_tracked_trader_opportunities(
        limit=50,
        include_filtered=False,
    )
    loader.assert_awaited_once_with(limit=50, include_filtered=False)
    assert result == rows


@pytest.mark.asyncio
async def test_load_tracked_trader_opportunities_include_filtered_passthrough(monkeypatch):
    rows = [{"id": "sig-filtered", "is_tradeable": False}]
    loader = AsyncMock(return_value=rows)
    monkeypatch.setattr(
        routes_discovery.StrategySDK,
        "get_trader_strategy_signals",
        loader,
    )
    result = await routes_discovery._load_tracked_trader_opportunities(
        limit=25,
        include_filtered=True,
    )
    loader.assert_awaited_once_with(limit=25, include_filtered=True)
    assert result == rows


@pytest.mark.asyncio
async def test_get_traders_overview_uses_strategy_filtered_rows(monkeypatch):
    confluence_rows = [
        {
            "id": "sig-traders",
            "market_id": "0xmarket",
            "market_question": "Will this be returned?",
            "wallets": ["0xabc"],
            "signal_type": "multi_wallet_buy",
            "outcome": "YES",
            "yes_price": 0.6,
            "no_price": 0.4,
            "detected_at": datetime.utcnow().isoformat(),
        }
    ]
    monkeypatch.setattr(routes_discovery.wallet_tracker, "get_all_wallets", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        routes_discovery,
        "_load_tracked_trader_opportunities",
        AsyncMock(return_value=confluence_rows),
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
    monkeypatch.setattr(
        routes_discovery,
        "_attach_live_mid_prices_to_signal_rows",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        routes_discovery,
        "_annotate_trader_signal_rows",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        routes_discovery,
        "_fetch_group_payload",
        AsyncMock(return_value=[]),
    )

    session = SimpleNamespace(execute=AsyncMock(return_value=_RowsResult([])))
    payload = await routes_discovery.get_traders_overview(
        tracked_limit=25,
        confluence_limit=10,
        hours=24,
        session=session,
    )
    assert payload["confluence"]["total_signals"] == 1
    assert payload["confluence"]["signals"][0]["id"] == "sig-traders"
    assert "min_tier" not in payload["confluence"]


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
