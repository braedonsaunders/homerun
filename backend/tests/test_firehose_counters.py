"""Tests for the per-gate firehose counters exposed at /metrics.

Firehose history is bounded and loss-tolerant, so monotonic counters are
the durable record of how the gate funnel performs.  These pin that the
emit helpers bump the right (event_type, strategy, gate, outcome) key.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.strategies import _firehose


async def _noop(*args, **kwargs):  # noqa: ARG001
    return None


def _arm(monkeypatch):
    """Force emission to fire and stub the durable write."""
    _firehose._firehose_counters.clear()

    async def _should_fire(strategy_slug):  # noqa: ARG001
        return True, ["trader-1"]

    monkeypatch.setattr(_firehose, "_emit_should_fire", _should_fire)
    monkeypatch.setattr(_firehose, "buffer_trader_event", _noop)


@pytest.mark.asyncio
async def test_emit_gate_bumps_counter(monkeypatch):
    _arm(monkeypatch)
    gate = _firehose.GateResult(name="min_distance", label="Min distance", passed=False, score=3.0)
    await _firehose.emit_gate(strategy_slug="Spike_Reversion", market={"slug": "btc"}, gate=gate)

    counters = _firehose.get_firehose_counters()
    # strategy is lowercased; rejected gate counted by name.
    assert counters.get("firehose_gate|spike_reversion|min_distance|rejected") == 1


@pytest.mark.asyncio
async def test_emit_evaluation_counts_failing_gate(monkeypatch):
    _arm(monkeypatch)
    gates = [
        _firehose.GateResult(name="timeframe", label="Timeframe", passed=True),
        _firehose.GateResult(name="oracle_fresh", label="Oracle freshness", passed=False),
    ]
    await _firehose.emit_evaluation(
        strategy_slug="momentum",
        market={"slug": "eth"},
        gates=gates,
        outcome="rejected",
    )

    counters = _firehose.get_firehose_counters()
    assert counters.get("firehose_evaluation|momentum|oracle_fresh|rejected") == 1


@pytest.mark.asyncio
async def test_emit_emit_counts_emitted(monkeypatch):
    _arm(monkeypatch)
    await _firehose.emit_emit(strategy_slug="momentum", market={"slug": "eth"}, detail="passed all")

    counters = _firehose.get_firehose_counters()
    assert counters.get("firehose_emit|momentum|-|emitted") == 1
