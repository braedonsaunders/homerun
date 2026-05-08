from __future__ import annotations

import sys
from pathlib import Path

import pytest
from sqlalchemy.dialects import postgresql

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services import trader_hot_state


def _reset_hot_state() -> None:
    trader_hot_state._audit_buffer.clear()


@pytest.mark.asyncio
async def test_update_buffered_decision_merges_pending_decision_updates():
    _reset_hot_state()
    try:
        decision_id = await trader_hot_state.buffer_decision(
            trader_id="trader-1",
            signal_id="signal-1",
            signal_source="scanner",
            strategy_key="tail_end_carry",
            strategy_version=1,
            decision="selected",
            reason="Tail carry signal selected",
            score=17.0,
            trace_id=None,
            checks_summary={"count": 34},
            risk_snapshot={},
            payload={"source_key": "scanner"},
            publish=False,
        )

        updated = await trader_hot_state.update_buffered_decision(
            decision_id=decision_id,
            decision="skipped",
            reason="BUY pre-submit gate failed: not enough collateral balance/allowance.",
            payload_patch={
                "execution_status": "skipped",
                "execution_skip_reason": "BUY pre-submit gate failed: not enough collateral balance/allowance.",
            },
            checks_summary_patch={"count": 35},
        )

        assert updated is True
        decision_entries = [
            entry for entry in trader_hot_state._audit_buffer if entry.kind == "decision" and entry.payload["id"] == decision_id
        ]
        assert len(decision_entries) == 1
        payload = decision_entries[0].payload
        assert payload["decision"] == "skipped"
        assert payload["reason"] == "BUY pre-submit gate failed: not enough collateral balance/allowance."
        assert payload["checks_summary_json"] == {"count": 35}
        assert payload["payload_json"]["source_key"] == "scanner"
        assert payload["payload_json"]["execution_status"] == "skipped"
        assert payload["payload_json"]["execution_skip_reason"].startswith("BUY pre-submit gate failed")
    finally:
        _reset_hot_state()


@pytest.mark.asyncio
async def test_flush_audit_buffer_requeues_failed_batch_at_front(monkeypatch):
    _reset_hot_state()

    class _NoAutoflush:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    class _Session:
        def __init__(self):
            self.no_autoflush = _NoAutoflush()

        async def execute(self, *args, **kwargs):
            return None

        async def commit(self):
            raise TimeoutError()

        async def rollback(self):
            return None

    class _SessionContext:
        async def __aenter__(self):
            return _Session()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    original_batch_size = trader_hot_state._AUDIT_FLUSH_BATCH_SIZE
    monkeypatch.setattr(trader_hot_state, "_AUDIT_FLUSH_BATCH_SIZE", 2)
    monkeypatch.setattr(trader_hot_state, "AuditAsyncSessionLocal", lambda: _SessionContext())

    try:
        trader_hot_state._audit_buffer.extend(
            [
                trader_hot_state._AuditEntry(kind="consumption", payload={"id": "first"}, created_at=1.0),
                trader_hot_state._AuditEntry(kind="consumption", payload={"id": "second"}, created_at=2.0),
                trader_hot_state._AuditEntry(kind="consumption", payload={"id": "third"}, created_at=3.0),
            ]
        )

        flushed = await trader_hot_state.flush_audit_buffer()

        assert flushed == 0
        assert [entry.payload["id"] for entry in trader_hot_state._audit_buffer] == [
            "first",
            "second",
            "third",
        ]
        # Each re-queue increments the entry's retry_count so the
        # exhaustion path can drop entries after _AUDIT_MAX_RETRIES
        # cycles instead of looping forever.  Only the first two were
        # in this flush's batch (batch_size=2); "third" was untouched
        # and stays at retry_count=0.
        buffer_by_id = {e.payload["id"]: e for e in trader_hot_state._audit_buffer}
        assert buffer_by_id["first"].retry_count == 1
        assert buffer_by_id["second"].retry_count == 1
        assert buffer_by_id["third"].retry_count == 0
    finally:
        monkeypatch.setattr(trader_hot_state, "_AUDIT_FLUSH_BATCH_SIZE", original_batch_size)
        _reset_hot_state()


@pytest.mark.asyncio
async def test_flush_audit_buffer_drops_entries_that_exhaust_retry_budget(monkeypatch, caplog):
    """Entries that fail to flush ``_AUDIT_MAX_RETRIES + 1`` times must
    be dropped permanently rather than cycling forever.  Without this
    cap, a persistently-failing audit kind (e.g. a row whose lock is
    permanently held by another writer) would re-queue every 500ms,
    growing the buffer toward its 50K cap and silently dropping fresher
    entries via FIFO eviction — corrupting audit ordering invisibly.
    """
    _reset_hot_state()

    class _NoAutoflush:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    class _Session:
        def __init__(self):
            self.no_autoflush = _NoAutoflush()

        async def execute(self, *args, **kwargs):
            return None

        async def commit(self):
            raise TimeoutError("simulated lock timeout")

        async def rollback(self):
            return None

    class _SessionContext:
        async def __aenter__(self):
            return _Session()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(trader_hot_state, "_AUDIT_FLUSH_BATCH_SIZE", 5)
    monkeypatch.setattr(trader_hot_state, "AuditAsyncSessionLocal", lambda: _SessionContext())

    try:
        # Seed an entry already at one-below the cap.  After one more
        # failed flush it should be dropped permanently rather than
        # re-queued.
        primed_entry = trader_hot_state._AuditEntry(
            kind="consumption",
            payload={"id": "primed"},
            created_at=1.0,
        )
        primed_entry.retry_count = trader_hot_state._AUDIT_MAX_RETRIES
        trader_hot_state._audit_buffer.append(primed_entry)

        flushed = await trader_hot_state.flush_audit_buffer()
        assert flushed == 0

        # Buffer must be empty: the entry exhausted its retry budget on
        # this flush and was dropped, NOT re-queued.
        assert trader_hot_state._audit_buffer == []
    finally:
        _reset_hot_state()


@pytest.mark.asyncio
async def test_flush_consumptions_nulls_missing_decision_ids_before_upsert():
    class _Result:
        def scalars(self):
            return self

        def all(self):
            return ["decision-ok"]

    class _Session:
        def __init__(self):
            self.statements = []

        async def execute(self, stmt):
            self.statements.append(stmt)
            return _Result()

    session = _Session()

    await trader_hot_state._flush_consumptions_bulk(
        session,
        [
            {
                "trader_id": "trader-1",
                "signal_id": "signal-1",
                "decision_id": "decision-ok",
                "outcome": "selected",
            },
            {
                "trader_id": "trader-1",
                "signal_id": "signal-2",
                "decision_id": "decision-missing",
                "outcome": "failed",
            },
        ],
    )

    assert len(session.statements) == 2
    compiled = session.statements[1].compile(dialect=postgresql.dialect())
    assert compiled.params["decision_id_m0"] == "decision-ok"
    assert compiled.params["decision_id_m1"] is None
    assert "coalesce(excluded.decision_id, trader_signal_consumption.decision_id)" in str(compiled).lower()


@pytest.mark.asyncio
async def test_flush_decision_checks_filters_missing_decision_ids():
    class _Result:
        def scalars(self):
            return self

        def all(self):
            return ["decision-ok"]

    class _Session:
        def __init__(self):
            self.statements = []

        async def execute(self, stmt):
            self.statements.append(stmt)
            return _Result()

    session = _Session()

    await trader_hot_state._flush_decision_checks_bulk(
        session,
        [
            {"decision_id": "decision-ok", "checks": [{"key": "risk", "label": "Risk", "passed": True}]},
            {"decision_id": "decision-missing", "checks": [{"key": "risk", "label": "Risk", "passed": False}]},
        ],
    )

    assert len(session.statements) == 2
    compiled = session.statements[1].compile(dialect=postgresql.dialect())
    assert compiled.params["decision_id_m0"] == "decision-ok"
    assert "decision_id_m1" not in compiled.params
