from __future__ import annotations

import sys
from pathlib import Path

import pytest

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
    monkeypatch.setattr(trader_hot_state, "AsyncSessionLocal", lambda: _SessionContext())

    try:
        trader_hot_state._audit_buffer.extend(
            [
                trader_hot_state._AuditEntry(kind="first", payload={"id": "first"}, created_at=1.0),
                trader_hot_state._AuditEntry(kind="second", payload={"id": "second"}, created_at=2.0),
                trader_hot_state._AuditEntry(kind="third", payload={"id": "third"}, created_at=3.0),
            ]
        )

        flushed = await trader_hot_state.flush_audit_buffer()

        assert flushed == 0
        assert [entry.payload["id"] for entry in trader_hot_state._audit_buffer] == [
            "first",
            "second",
            "third",
        ]
    finally:
        monkeypatch.setattr(trader_hot_state, "_AUDIT_FLUSH_BATCH_SIZE", original_batch_size)
        _reset_hot_state()
