import asyncio
import sys
from datetime import timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from models.database import OpportunityState, ScannerRun, ScannerSnapshot  # noqa: E402
from models.opportunity import MispricingType, Opportunity  # noqa: E402
from services import shared_state  # noqa: E402
from utils.utcnow import utcnow  # noqa: E402


class _FakeScalarResult:
    def __init__(self, scalar_value=None, scalars_list=None, rowcount=None):
        self._scalar_value = scalar_value
        self._scalars_list = list(scalars_list or [])
        self.rowcount = rowcount

    def scalar_one_or_none(self):
        return self._scalar_value

    def scalar_one(self):
        return self._scalar_value

    def one_or_none(self):
        return self._scalar_value

    def scalars(self):
        return self

    def all(self):
        return list(self._scalars_list)


class _FakeSession:
    def __init__(self, *, snapshot_row=None, existing_rows=None, execute_results=None):
        self.snapshot_row = snapshot_row
        self.existing_rows = list(existing_rows or [])
        self.execute_results = list(execute_results or [])
        self.added = []
        self.execute_calls = 0
        self.statements = []

    async def execute(self, statement, params=None):
        self.statements.append((statement, params))
        sql_text = str(statement)
        if "SET LOCAL statement_timeout" in sql_text or "SET LOCAL lock_timeout" in sql_text:
            return _FakeScalarResult()
        if self.execute_results:
            result = self.execute_results.pop(0)
            if isinstance(result, _FakeScalarResult):
                return result
            return _FakeScalarResult(
                scalar_value=result.get("scalar_value"),
                scalars_list=result.get("scalars_list"),
                rowcount=result.get("rowcount"),
            )
        self.execute_calls += 1
        if self.execute_calls == 1:
            return _FakeScalarResult(scalar_value=self.snapshot_row)
        return _FakeScalarResult(scalars_list=self.existing_rows)

    def add(self, value):
        if isinstance(value, ScannerSnapshot):
            self.snapshot_row = value
        self.added.append(value)


def _build_opportunity(*, market_id: str) -> Opportunity:
    now = utcnow().replace(tzinfo=timezone.utc)
    return Opportunity(
        strategy="generic_strategy",
        title=f"Opportunity {market_id}",
        description="scanner opportunity",
        total_cost=0.91,
        expected_payout=1.0,
        gross_profit=0.09,
        fee=0.01,
        net_profit=0.08,
        roi_percent=8.79,
        risk_score=0.2,
        risk_factors=["unit_test"],
        markets=[
            {
                "id": market_id,
                "condition_id": market_id,
                "question": f"Question {market_id}",
                "yes_price": 0.09,
                "no_price": 0.91,
                "liquidity": 25000,
                "price_history": [
                    {"t": 1, "yes": 0.1, "no": 0.9},
                    {"t": 2, "yes": 0.09, "no": 0.91},
                ],
            }
        ],
        positions_to_take=[
            {
                "action": "BUY",
                "outcome": "NO",
                "price": 0.91,
                "token_id": f"token-{market_id}",
            }
        ],
        event_id=f"evt-{market_id}",
        event_title=f"Event {market_id}",
        category="Sports",
        min_liquidity=25000.0,
        max_position_size=2500.0,
        detected_at=now,
        mispricing_type=MispricingType.SETTLEMENT_LAG,
    )


@pytest.mark.asyncio
async def test_write_scanner_snapshot_publishes_runtime_events_without_db_opportunity_event_inserts(monkeypatch):
    publish_mock = AsyncMock()
    commit_mock = AsyncMock()
    schedule_mock = Mock()
    history_upsert_mock = AsyncMock(return_value=0)
    monkeypatch.setattr(shared_state.event_bus, "publish", publish_mock)
    monkeypatch.setattr(shared_state, "_commit_with_retry", commit_mock)
    monkeypatch.setattr(shared_state, "_schedule_scanner_state_projection", schedule_mock)
    monkeypatch.setattr(shared_state, "upsert_scanner_market_history", history_upsert_mock)

    session = _FakeSession()
    opportunity = _build_opportunity(market_id="market-1")
    status = {
        "running": True,
        "enabled": True,
        "interval_seconds": 60,
        "current_activity": "Fast scan complete - 1 found, 1 total",
        "last_scan": utcnow(),
        "strategies": [{"name": "Generic Strategy", "type": "generic_strategy"}],
    }

    await shared_state.write_scanner_snapshot(session, [opportunity], status)
    await asyncio.sleep(0)

    assert isinstance(session.snapshot_row, ScannerSnapshot)
    assert session.snapshot_row.opportunities_count == 1
    assert list(session.snapshot_row.opportunities_json or []) == []
    commit_mock.assert_awaited_once()
    schedule_mock.assert_called_once()
    scheduled = schedule_mock.call_args.kwargs["opportunities"]
    assert len(scheduled) == 1
    assert scheduled[0].stable_id == opportunity.stable_id
    history_upsert_mock.assert_not_awaited()

    published_types = [call.args[0] for call in publish_mock.await_args_list]
    assert "scanner_status" in published_types
    assert "scanner_activity" in published_types
    assert "opportunities_update" in published_types
    assert "opportunity_events" not in published_types
    assert "opportunity_update" not in published_types


@pytest.mark.asyncio
async def test_persist_incremental_state_emits_detected_event_via_batched_upsert():
    """New opportunity → "detected" event + a batched pg_insert UPSERT.

    The previous implementation called ``session.add(OpportunityState(...))``
    per row, which generated one INSERT statement per row at flush.  The
    new implementation issues a single ``INSERT ... ON CONFLICT DO UPDATE``
    per chunk of 500 rows, so OpportunityState should NOT appear in
    session.added — only ScannerRun does.
    """
    opportunity = _build_opportunity(market_id="market-2")
    payload = [opportunity.model_dump(mode="json")]
    completed_at = utcnow().replace(tzinfo=None)
    # Sequence: _load_existing_meta (empty) → active_ids (empty) → upsert execute.
    session = _FakeSession(execute_results=[{"scalars_list": []}, {"scalars_list": []}])

    event_messages = await shared_state._persist_incremental_state(
        session,
        payload,
        {"current_activity": "Fast scan complete - 1 found, 1 total"},
        completed_at,
    )

    added_types = tuple(type(row) for row in session.added)
    assert ScannerRun in added_types
    # OpportunityState rows go through pg_insert, not session.add.
    assert OpportunityState not in added_types
    assert len(event_messages) == 1
    assert event_messages[0]["event_type"] == "detected"
    assert event_messages[0]["stable_id"] == opportunity.stable_id
    # An INSERT ... ON CONFLICT statement was executed for the row.
    insert_statements = [
        sql for sql, _ in session.statements if "INSERT INTO opportunity_state" in str(sql)
    ]
    assert insert_statements, "expected a batched pg_insert into opportunity_state"


@pytest.mark.asyncio
async def test_persist_incremental_state_emits_reactivated_event_for_inactive_existing_row():
    """Existing inactive opportunity reappearing → "reactivated" event + UPSERT."""
    import json as _json

    opportunity = _build_opportunity(market_id="market-4")
    payload = [opportunity.model_dump(mode="json")]
    completed_at = utcnow().replace(tzinfo=None)
    # The new implementation reads (stable_id, is_active, first_seen_at,
    # opportunity_json_text) tuples — opportunity_json is text-cast so
    # the fake returns a JSON string, not a dict.
    existing_meta_row = (
        opportunity.stable_id,
        False,  # is_active
        utcnow().replace(tzinfo=None),
        _json.dumps(opportunity.model_dump(mode="json")),
    )
    session = _FakeSession(
        execute_results=[
            {"scalars_list": [existing_meta_row]},  # _load_existing_meta
            {"scalars_list": []},                    # active_ids select (no actives)
        ]
    )

    event_messages = await shared_state._persist_incremental_state(
        session,
        payload,
        {"current_activity": "Fast scan complete - 1 found, 1 total"},
        completed_at,
    )

    assert len(event_messages) == 1
    assert event_messages[0]["event_type"] == "reactivated"
    assert event_messages[0]["stable_id"] == opportunity.stable_id
    insert_statements = [
        sql for sql, _ in session.statements if "INSERT INTO opportunity_state" in str(sql)
    ]
    assert insert_statements, "expected a batched pg_insert UPSERT for reactivation"


@pytest.mark.asyncio
async def test_persist_incremental_state_emits_expired_event_via_bulk_update():
    """Active row missing from current scan → "expired" event + bulk UPDATE.

    The previous implementation mutated the loaded ORM row's is_active
    attribute, generating a per-row UPDATE at flush.  The new
    implementation issues ``UPDATE ... WHERE stable_id IN (...)
    AND is_active = TRUE`` once per chunk, so we no longer touch ORM
    instances at all.
    """
    import json as _json

    opportunity = _build_opportunity(market_id="market-3")
    completed_at = utcnow().replace(tzinfo=None)
    # Sequence (note: _load_existing_meta short-circuits on empty
    # stable_ids and does NOT issue a SELECT, so its result is omitted):
    #   1. active_ids select — returns the stable_id of the row to expire
    #   2. expired-payload-text select — returns (stable_id, json_text) tuple
    #   3. bulk UPDATE — execute call (no result needed)
    session = _FakeSession(
        execute_results=[
            {"scalars_list": [opportunity.stable_id]},
            {
                "scalars_list": [
                    (opportunity.stable_id, _json.dumps(opportunity.model_dump(mode="json"))),
                ]
            },
        ]
    )

    event_messages = await shared_state._persist_incremental_state(
        session,
        [],
        {"current_activity": "Fast scan complete - 0 found, 0 total"},
        completed_at,
    )

    assert len(event_messages) == 1
    assert event_messages[0]["event_type"] == "expired"
    assert event_messages[0]["stable_id"] == opportunity.stable_id
    update_statements = [
        sql for sql, _ in session.statements if "UPDATE opportunity_state" in str(sql)
    ]
    assert update_statements, "expected a bulk UPDATE setting is_active=false for expirations"


@pytest.mark.asyncio
async def test_write_market_catalog_serializes_before_db_checkout(monkeypatch):
    commit_mock = AsyncMock()
    monkeypatch.setattr(shared_state, "_commit_with_retry", commit_mock)
    session = _FakeSession(execute_results=[{"rowcount": 1}])

    class _ProbeEvent:
        def model_dump(self, *, mode="json", exclude=None):
            assert session.statements == []
            payload = {
                "id": "event-1",
                "slug": "event-1",
                "title": "Event One",
                "markets": [{"id": "market-1"}],
            }
            for key in list(exclude or set()):
                payload.pop(key, None)
            return payload

    class _ProbeMarket:
        def model_dump(self, *, mode="json"):
            assert session.statements == []
            return {
                "id": "market-1",
                "condition_id": "condition-1",
                "question": "Will team A win?",
            }

    await shared_state.write_market_catalog(
        session,
        [_ProbeEvent()],
        [_ProbeMarket()],
        duration_seconds=1.5,
    )

    commit_mock.assert_awaited_once()
    assert any("UPDATE" in str(statement) for statement, _ in session.statements)


@pytest.mark.asyncio
async def test_write_traders_snapshot_serializes_before_db_checkout(monkeypatch):
    commit_mock = AsyncMock()
    monkeypatch.setattr(shared_state, "_commit_with_retry", commit_mock)
    session = _FakeSession()

    class _ProbeOpportunity:
        stable_id = "opp-1"

        def model_dump(self, *, mode="json"):
            assert session.statements == []
            return {
                "id": "opp-1",
                "stable_id": "opp-1",
                "strategy": "generic_strategy",
                "title": "Opportunity",
                "strategy_context": {},
            }

    await shared_state.write_traders_snapshot(
        session,
        [_ProbeOpportunity()],
        {"running": True, "enabled": True, "current_activity": "Testing"},
    )

    commit_mock.assert_awaited_once()
    assert isinstance(session.snapshot_row, ScannerSnapshot)
    assert session.snapshot_row.opportunities_json[0]["strategy_context"]["source_key"] == "traders"


@pytest.mark.asyncio
async def test_read_scanner_snapshot_reads_active_opportunities_from_state_rows():
    import json as _json

    opportunity = _build_opportunity(market_id="market-9")
    payload = opportunity.model_dump(mode="json")
    # ``opportunity_json`` is now selected as ``cast(... AS text)`` so
    # asyncpg does not parse it on the event loop.  The fake session
    # therefore returns the JSON-encoded text the way the real DB does.
    payload_text = _json.dumps(payload)
    snapshot_row = SimpleNamespace(
        running=True,
        enabled=True,
        interval_seconds=60,
        last_scan_at=utcnow(),
        current_activity="Fast scan complete - 1 found, 1 total",
        strategies_json=[{"name": "Generic Strategy", "type": "generic_strategy"}],
        strategy_diagnostics_json={},
        tiered_scanning_json={},
        ws_feeds_json={},
        opportunities_count=1,
    )
    # Same applies to ``points_json`` — read_scanner_market_history now
    # selects the column as text for off-loop decode.
    points_text = _json.dumps(
        [
            {"t": 1.0, "yes": 0.11, "no": 0.89},
            {"t": 2.0, "yes": 0.09, "no": 0.91},
        ]
    )
    session = _FakeSession(
        execute_results=[
            {"scalar_value": snapshot_row},
            {"scalars_list": [payload_text]},
            {
                "scalars_list": [
                    ("market-9", points_text),
                ]
            },
        ]
    )

    opportunities, status = await shared_state.read_scanner_snapshot(session)

    assert len(opportunities) == 1
    assert len(opportunities[0].markets[0].get("price_history") or []) == 2
    assert status["opportunities_count"] == 1
