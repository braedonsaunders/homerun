import asyncio
import sys
from pathlib import Path
from datetime import date, datetime, timedelta, timezone

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from models.opportunity import ArbitrageOpportunity, StrategyType
from services import market_tradability, shared_state
from services.weather import shared_state as weather_shared_state


def _opp(market_id: str) -> ArbitrageOpportunity:
    return ArbitrageOpportunity(
        strategy=StrategyType.BASIC,
        title=f"Opp {market_id}",
        description="test",
        total_cost=0.9,
        expected_payout=1.0,
        gross_profit=0.1,
        fee=0.0,
        net_profit=0.1,
        roi_percent=10.0,
        markets=[{"id": market_id}],
        positions_to_take=[{"market_id": market_id, "outcome": "YES", "price": 0.45}],
    )


def _report_only_opp(market_id: str) -> ArbitrageOpportunity:
    return ArbitrageOpportunity(
        strategy=StrategyType.WEATHER_EDGE,
        title=f"Report {market_id}",
        description="report only",
        total_cost=0.0,
        expected_payout=0.0,
        gross_profit=0.0,
        fee=0.0,
        net_profit=0.0,
        roi_percent=0.0,
        markets=[{"id": market_id}],
        positions_to_take=[],
        max_position_size=0.0,
    )


def test_scanner_opportunities_filtered_by_market_tradability(monkeypatch):
    good = _opp("0xgood")
    bad = _opp("0xbad")

    async def _fake_read(_session):
        return [good, bad], {}

    async def _fake_map(market_ids, **_kwargs):
        return {str(mid).lower(): str(mid).lower() != "0xbad" for mid in market_ids}

    monkeypatch.setattr(shared_state, "read_scanner_snapshot", _fake_read)
    monkeypatch.setattr(shared_state, "get_market_tradability_map", _fake_map)

    rows = asyncio.run(shared_state.get_opportunities_from_db(session=None, filter=None))
    assert [r.markets[0]["id"] for r in rows] == ["0xgood"]


def test_weather_opportunities_filtered_by_market_tradability(monkeypatch):
    good = _opp("0xweathergood")
    bad = _opp("0xweatherbad")

    async def _fake_read(_session):
        return [good, bad], {}

    async def _fake_map(market_ids, **_kwargs):
        return {
            str(mid).lower(): str(mid).lower() != "0xweatherbad"
            for mid in market_ids
        }

    monkeypatch.setattr(weather_shared_state, "read_weather_snapshot", _fake_read)
    monkeypatch.setattr(weather_shared_state, "get_market_tradability_map", _fake_map)

    rows = asyncio.run(
        weather_shared_state.get_weather_opportunities_from_db(
            session=None, require_tradable_markets=True
        )
    )
    assert [r.markets[0]["id"] for r in rows] == ["0xweathergood"]


def test_weather_opportunities_drop_near_resolution(monkeypatch):
    soon = _opp("0xsoon")
    soon.resolution_date = datetime.now(timezone.utc) + timedelta(minutes=10)
    later = _opp("0xlater")
    later.resolution_date = datetime.now(timezone.utc) + timedelta(hours=3)

    async def _fake_read(_session):
        return [soon, later], {}

    async def _fake_map(market_ids, **_kwargs):
        return {str(mid).lower(): True for mid in market_ids}

    monkeypatch.setattr(weather_shared_state, "read_weather_snapshot", _fake_read)
    monkeypatch.setattr(weather_shared_state, "get_market_tradability_map", _fake_map)

    rows = asyncio.run(
        weather_shared_state.get_weather_opportunities_from_db(
            session=None, exclude_near_resolution=True
        )
    )
    assert [r.markets[0]["id"] for r in rows] == ["0xlater"]


def test_weather_opportunities_default_keeps_reports(monkeypatch):
    soon = _opp("0xsoon")
    soon.resolution_date = datetime.now(timezone.utc) + timedelta(minutes=10)
    untradable = _opp("0xweatherbad")

    async def _fake_read(_session):
        return [soon, untradable], {}

    async def _fake_map(market_ids, **_kwargs):
        return {str(mid).lower(): str(mid).lower() != "0xweatherbad" for mid in market_ids}

    monkeypatch.setattr(weather_shared_state, "read_weather_snapshot", _fake_read)
    monkeypatch.setattr(weather_shared_state, "get_market_tradability_map", _fake_map)

    rows = asyncio.run(weather_shared_state.get_weather_opportunities_from_db(session=None))
    assert [r.markets[0]["id"] for r in rows] == ["0xsoon", "0xweatherbad"]


def test_weather_max_entry_keeps_report_only_cards(monkeypatch):
    report_only = _report_only_opp("0xreport")
    expensive = _opp("0xexpensive")

    async def _fake_read(_session):
        return [report_only, expensive], {}

    async def _fake_map(_market_ids, **_kwargs):
        return {}

    monkeypatch.setattr(weather_shared_state, "read_weather_snapshot", _fake_read)
    monkeypatch.setattr(weather_shared_state, "get_market_tradability_map", _fake_map)

    rows = asyncio.run(
        weather_shared_state.get_weather_opportunities_from_db(
            session=None, max_entry_price=0.25
        )
    )
    assert [r.markets[0]["id"] for r in rows] == ["0xreport"]


def test_weather_opportunities_filtered_by_target_date(monkeypatch):
    feb13 = _opp("0xfeb13")
    feb13.markets[0]["weather"] = {"target_time": "2026-02-13T07:00:00Z"}
    feb14 = _opp("0xfeb14")
    feb14.markets[0]["weather"] = {"target_time": "2026-02-14T07:00:00Z"}
    fallback = _opp("0xfallback")
    fallback.resolution_date = datetime(2026, 2, 13, 19, 0, tzinfo=timezone.utc)

    async def _fake_read(_session):
        return [feb13, feb14, fallback], {}

    async def _fake_map(_market_ids, **_kwargs):
        return {}

    monkeypatch.setattr(weather_shared_state, "read_weather_snapshot", _fake_read)
    monkeypatch.setattr(weather_shared_state, "get_market_tradability_map", _fake_map)

    rows = asyncio.run(
        weather_shared_state.get_weather_opportunities_from_db(
            session=None,
            target_date=date(2026, 2, 13),
        )
    )
    assert [r.markets[0]["id"] for r in rows] == ["0xfeb13", "0xfallback"]


def test_weather_target_date_counts_from_snapshot(monkeypatch):
    feb13_a = _opp("0xfeb13a")
    feb13_a.markets[0]["weather"] = {"target_time": "2026-02-13T07:00:00Z"}
    feb13_b = _opp("0xfeb13b")
    feb13_b.resolution_date = datetime(2026, 2, 13, 22, 0, tzinfo=timezone.utc)
    feb14 = _opp("0xfeb14")
    feb14.markets[0]["weather"] = {"target_time": "2026-02-14T07:00:00Z"}

    async def _fake_read(_session):
        return [feb13_a, feb13_b, feb14], {}

    async def _fake_map(_market_ids, **_kwargs):
        return {}

    monkeypatch.setattr(weather_shared_state, "read_weather_snapshot", _fake_read)
    monkeypatch.setattr(weather_shared_state, "get_market_tradability_map", _fake_map)

    rows = asyncio.run(
        weather_shared_state.get_weather_target_date_counts_from_db(session=None)
    )
    assert rows == [
        {"date": "2026-02-13", "count": 2},
        {"date": "2026-02-14", "count": 1},
    ]


def test_weather_target_date_counts_fallback_to_question_text(monkeypatch):
    text_only = _opp("0xtext")
    text_only.markets[0]["question"] = (
        "Will the highest temperature in Wellington be 23C or higher on Feb 13?"
    )
    text_only.markets[0].pop("weather", None)
    text_only.resolution_date = None

    async def _fake_read(_session):
        return [text_only], {}

    async def _fake_map(_market_ids, **_kwargs):
        return {}

    monkeypatch.setattr(weather_shared_state, "read_weather_snapshot", _fake_read)
    monkeypatch.setattr(weather_shared_state, "get_market_tradability_map", _fake_map)

    rows = asyncio.run(
        weather_shared_state.get_weather_target_date_counts_from_db(session=None)
    )
    assert rows and rows[0]["count"] == 1


def test_market_tradability_map_handles_lookup_failure(monkeypatch):
    market_tradability._cache.clear()

    async def _raise_lookup(_market_id):
        raise RuntimeError("network down")

    monkeypatch.setattr(
        market_tradability.polymarket_client,
        "get_market_by_condition_id",
        _raise_lookup,
    )
    monkeypatch.setattr(
        market_tradability.polymarket_client,
        "get_market_by_token_id",
        _raise_lookup,
    )

    result = asyncio.run(
        market_tradability.get_market_tradability_map(["0xabc", "123"])
    )

    # Guard is fail-open for unknown lookups so we do not drop good markets on transient API errors.
    assert result["0xabc"] is True
    assert result["123"] is True


def test_market_tradability_map_skips_non_polymarket_ids(monkeypatch):
    market_tradability._cache.clear()

    async def _unexpected_lookup(_market_id):
        raise AssertionError("Polymarket lookup should not be called for non-Polymarket IDs")

    monkeypatch.setattr(
        market_tradability.polymarket_client,
        "get_market_by_condition_id",
        _unexpected_lookup,
    )
    monkeypatch.setattr(
        market_tradability.polymarket_client,
        "get_market_by_token_id",
        _unexpected_lookup,
    )

    ids = [
        "kxmvesportsmultigameextended-s202609d9e02ecb6-3f161df43ea",
        "kasimpasa vs karagumruk winner?",
        "KXINXSPXW-26FEB11-B5910_yes",
    ]
    result = asyncio.run(market_tradability.get_market_tradability_map(ids))

    for market_id in ids:
        assert result[market_id.lower()] is True


def test_market_tradability_map_routes_condition_and_token_ids(monkeypatch):
    market_tradability._cache.clear()
    calls = {"condition": 0, "token": 0}

    async def _condition_lookup(_market_id):
        calls["condition"] += 1
        return None

    async def _token_lookup(_market_id):
        calls["token"] += 1
        return None

    monkeypatch.setattr(
        market_tradability.polymarket_client,
        "get_market_by_condition_id",
        _condition_lookup,
    )
    monkeypatch.setattr(
        market_tradability.polymarket_client,
        "get_market_by_token_id",
        _token_lookup,
    )

    condition_id = "0x" + ("a" * 64)
    token_id = "1234567890123456789012345"
    result = asyncio.run(
        market_tradability.get_market_tradability_map([condition_id, token_id])
    )

    assert result[condition_id] is True
    assert result[token_id] is True
    assert calls == {"condition": 1, "token": 1}
