import sys
from datetime import datetime
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.data_source_sdk import BaseDataSource
from services.search import collectors


class _FakeResult:
    rowcount = 0


class _FakeSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object | None]] = []

    async def execute(self, _statement: object, params: object | None = None) -> _FakeResult:
        self.calls.append(("execute", params))
        return _FakeResult()

    async def commit(self) -> None:
        self.calls.append(("commit", None))

    async def rollback(self) -> None:
        self.calls.append(("rollback", None))


@pytest.mark.asyncio
async def test_reindex_one_releases_collector_read_transaction_before_writes():
    async def collect_unit_rows(session: _FakeSession) -> list[dict[str, object]]:
        session.calls.append(("collector", None))
        return [
            {
                "entity_id": "row-1",
                "title": "Search row",
                "subtitle": "unit",
                "body": "body",
                "category": "test",
                "tags": ["unit"],
                "metadata": {"source": "test"},
                "liquidity": 1.0,
                "volume": 2.0,
                "recency": datetime(2026, 5, 11, 12, 0),
            }
        ]

    collectors.COLLECTORS["unit_test"] = collect_unit_rows
    try:
        session = _FakeSession()
        result = await collectors.reindex_one(session, "unit_test")
    finally:
        collectors.COLLECTORS.pop("unit_test", None)

    assert result["ok"] is True
    assert result["upserted"] == 1

    call_names = [name for name, _payload in session.calls]
    assert call_names.index("collector") < call_names.index("rollback")
    assert call_names.index("rollback") < call_names.index("execute")


def test_base_data_source_parse_datetime_returns_naive_utc_for_comparisons():
    source = BaseDataSource()

    parsed = source._parse_datetime("2026-05-11T10:30:00-04:00")

    assert parsed == datetime(2026, 5, 11, 14, 30)
    assert parsed.tzinfo is None
    assert parsed < datetime(2026, 5, 11, 15, 0)
