"""Tests for the institutional firehose data-class split.

Firehose telemetry (gate evaluations / rejections / emits) must NEVER
persist to Postgres — it is ephemeral, loss-tolerant observability that
lives only in Redis: pub/sub for the live terminal feed plus a capped
Stream for bounded reload history.  Business-of-record events
(decisions, orders, errors) still flow through the durable audit buffer.

These tests pin that contract:
  * firehose events skip the Postgres audit buffer,
  * firehose events are routed to the capped Redis Stream (XADD),
  * non-firehose events still hit the audit buffer and never touch the
    stream,
  * ``safe_xrevrange`` decodes stream entries and soft-fails to ``[]``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services import redis_client, trader_hot_state


class _DummyClient:
    """Truthy stand-in so ``buffer_trader_event`` enters the publish branch."""


def _install_capture(monkeypatch):
    """Capture ``_schedule_trader_event_publish`` calls; stub Redis helpers.

    Returns the list that accumulates ``(channel, payload, kwargs)`` so
    the pump never runs and we can inspect routing decisions directly.
    """
    captured: list[tuple[str, str, dict]] = []

    def _fake_schedule(client, channel, payload, *, stream=None, stream_maxlen=None):
        captured.append((channel, payload, {"stream": stream, "stream_maxlen": stream_maxlen}))

    monkeypatch.setattr(redis_client, "get_client_or_none", lambda: _DummyClient())
    monkeypatch.setattr(redis_client, "namespaced", lambda key: f"homerun:{key}")
    monkeypatch.setattr(trader_hot_state, "_schedule_trader_event_publish", _fake_schedule)
    return captured


@pytest.mark.asyncio
async def test_firehose_event_skips_audit_buffer_and_routes_to_stream(monkeypatch):
    trader_hot_state._audit_buffer.clear()
    captured = _install_capture(monkeypatch)

    await trader_hot_state.buffer_trader_event(
        event_type="firehose_gate",
        severity="info",
        verbosity="murmur",
        source="crypto",
        message="gate rejected",
        payload={"source_key": "crypto", "strategy_slug": "spike_reversion"},
    )

    # Firehose telemetry must NOT persist to Postgres.
    trader_events = [e for e in trader_hot_state._audit_buffer if e.kind == "trader_event"]
    assert trader_events == []

    # It must be routed to the capped firehose stream (one publish call).
    assert len(captured) == 1
    _channel, payload, kwargs = captured[0]
    assert kwargs["stream"] == f"homerun:{trader_hot_state._FIREHOSE_STREAM}"
    assert kwargs["stream_maxlen"] == trader_hot_state._FIREHOSE_STREAM_MAXLEN
    assert "firehose_gate" in payload


@pytest.mark.asyncio
async def test_all_firehose_event_types_skip_postgres(monkeypatch):
    for event_type in sorted(trader_hot_state._FIREHOSE_EVENT_TYPES):
        trader_hot_state._audit_buffer.clear()
        captured = _install_capture(monkeypatch)
        await trader_hot_state.buffer_trader_event(
            event_type=event_type,
            source="crypto",
            payload={"source_key": "crypto"},
        )
        assert [e for e in trader_hot_state._audit_buffer if e.kind == "trader_event"] == [], event_type
        assert captured[0][2]["stream"] == f"homerun:{trader_hot_state._FIREHOSE_STREAM}", event_type


@pytest.mark.asyncio
async def test_non_firehose_event_persists_and_skips_stream(monkeypatch):
    trader_hot_state._audit_buffer.clear()
    captured = _install_capture(monkeypatch)

    await trader_hot_state.buffer_trader_event(
        event_type="decision",
        severity="info",
        trader_id="trader-1",
        source="scanner",
        message="decision made",
        payload={"decision_id": "d1"},
    )

    # Business-of-record events still land in the durable audit buffer.
    trader_events = [e for e in trader_hot_state._audit_buffer if e.kind == "trader_event"]
    assert len(trader_events) == 1
    assert trader_events[0].payload["event_type"] == "decision"
    assert trader_events[0].payload["trader_id"] == "trader-1"

    # ...and are NOT written to the firehose stream.
    assert len(captured) == 1
    assert captured[0][2]["stream"] is None
    assert captured[0][2]["stream_maxlen"] is None


class _FakeRedisXRevRange:
    def __init__(self, rows):
        self._rows = rows

    async def xrevrange(self, name, count=500):  # noqa: ARG002
        return self._rows


@pytest.mark.asyncio
async def test_safe_xrevrange_decodes_bytes(monkeypatch):
    rows = [
        (b"2-0", {b"data": b'{"id":"b","event_type":"firehose_emit"}'}),
        (b"1-0", {b"data": b'{"id":"a","event_type":"firehose_gate"}'}),
    ]
    monkeypatch.setattr(redis_client, "get_client_or_none", lambda: _FakeRedisXRevRange(rows))

    out = await redis_client.safe_xrevrange("trader_firehose", count=10)

    assert out == [
        ("2-0", {"data": '{"id":"b","event_type":"firehose_emit"}'}),
        ("1-0", {"data": '{"id":"a","event_type":"firehose_gate"}'}),
    ]


@pytest.mark.asyncio
async def test_safe_xrevrange_soft_fails_without_client(monkeypatch):
    monkeypatch.setattr(redis_client, "get_client_or_none", lambda: None)
    assert await redis_client.safe_xrevrange("trader_firehose") == []
