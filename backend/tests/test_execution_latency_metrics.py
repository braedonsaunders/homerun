from __future__ import annotations

import pytest

from services.execution_latency_metrics import ExecutionLatencyMetrics


@pytest.mark.asyncio
async def test_execution_latency_snapshot_uses_rolling_window(monkeypatch):
    current_time = 1_000.0
    monkeypatch.setattr("services.execution_latency_metrics.time.time", lambda: current_time)

    metrics = ExecutionLatencyMetrics()

    await metrics.record(
        trader_id="trader-1",
        source="scanner",
        strategy_key="tail_end_carry",
        payload={
            "armed_to_ws_release_ms": 30_000,
            "ws_release_to_decision_ms": 600,
            "ws_release_to_submit_start_ms": 31_000,
            "emit_to_submit_start_ms": 31_000,
        },
    )

    current_time = 1_890.0
    await metrics.record(
        trader_id="trader-1",
        source="scanner",
        strategy_key="tail_end_carry",
        payload={
            "armed_to_ws_release_ms": 15,
            "ws_release_to_decision_ms": 40,
            "ws_release_to_submit_start_ms": 95,
            "emit_to_submit_start_ms": 95,
        },
    )

    current_time = 1_901.0
    snapshot = await metrics.snapshot()

    assert snapshot["internal_sla_definition"] == "ws_release_at->submit_started_at"
    assert snapshot["rolling_window_seconds"] == 900
    assert snapshot["sample_count"] == 1
    assert snapshot["overall"]["count"] == 1
    assert snapshot["overall"]["armed_to_ws_release_ms"]["p95"] == 15
    assert snapshot["overall"]["ws_release_to_decision_ms"]["p95"] == 40
    assert snapshot["overall"]["ws_release_to_submit_start_ms"]["p95"] == 95
    assert snapshot["by_source"]["scanner"]["ws_release_to_submit_start_ms"]["p95"] == 95
