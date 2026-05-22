"""Tests for Phase 1 of the gate-pipeline refactor.

The Phase 1 hoist moves the BUY collateral gate and the
``_check_max_spread_bps`` venue-book gate out of ``submit_leg`` (which
runs AFTER ``_commit_pre_submit_projection``) into a new
``pre_db_venue_preflight`` step that runs BEFORE the placeholder DB
writes.  Rejected signals no longer pay the 5-13s DB cost when the
collateral or spread gate would reject anyway.

The four scenarios below cover the contract:

1. Happy path: all legs pass preflight → placeholders are written and
   ``submit_execution_wave`` is invoked with the full wave.
2. All-reject spread: every leg fails the spread gate → projection
   commit is NOT called and every wave_result is status='skipped'.
3. Mixed: 1 leg passes, 2 fail → only the 1 passing leg's placeholder
   is written and only it goes through ``submit_execution_wave``.
4. BUY collateral fail: leg is marked status='skipped' with
   reason='buy_pre_submit_gate'.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.trader_orchestrator import session_engine as session_engine_module
from services.trader_orchestrator import venue_gates as venue_gates_module
from services import intent_runtime as intent_runtime_module


def _leg(
    *,
    leg_id: str,
    market_id: str,
    token_id: str,
    side: str = "buy",
    limit_price: float = 0.5,
    requested_notional_usd: float = 10.0,
    requested_shares: float = 20.0,
) -> dict:
    return {
        "leg_id": leg_id,
        "market_id": market_id,
        "market_question": "Will the test pass?",
        "token_id": token_id,
        "side": side,
        "outcome": "yes",
        "requested_notional_usd": requested_notional_usd,
        "requested_shares": requested_shares,
        "limit_price": limit_price,
        "price_policy": "taker_limit",
        "time_in_force": "IOC",
        "post_only": False,
    }


def _signal(*, signal_id: str, market_id: str, entry_price: float = 0.5) -> SimpleNamespace:
    return SimpleNamespace(
        id=signal_id,
        source="scanner",
        trace_id=f"trace-{signal_id}",
        strategy_type="generic_strategy",
        strategy_context_json={},
        payload_json={},
        market_id=market_id,
        market_question="Will the test pass?",
        direction="buy_yes",
        entry_price=entry_price,
        edge_percent=5.0,
        confidence=0.7,
    )


class _RecordingDb:
    """Minimal db double — records every flush/commit and the order rows
    that pass through.  Mirrors the simpler tests in
    test_execution_session_engine.py.

    ``commit_snapshots`` captures the status of every TraderOrder at the
    moment ``commit()`` is invoked.  This is the durable record we use
    in assertions, because the row objects are mutated in place between
    the pre-submit commit and the final post-submit commit, so reading
    their state later only sees the last status.
    """

    def __init__(self) -> None:
        self.pending: list[object] = []
        self.persisted_rows_by_type: dict[str, list[object]] = {}
        self.flush_calls = 0
        self.commit_calls = 0
        self.commit_snapshots: list[list[tuple[str, str]]] = []

    def add(self, row: object) -> None:
        self.pending.append(row)

    async def flush(self) -> None:
        self.flush_calls += 1
        for row in self.pending:
            row_type = row.__class__.__name__
            self.persisted_rows_by_type.setdefault(row_type, []).append(row)
        self.pending.clear()

    async def commit(self) -> None:
        self.commit_calls += 1
        # Snapshot the current TraderOrder statuses so we can prove
        # whether the pre-submit commit (status='placing') ran.
        trader_orders = self.persisted_rows_by_type.get("TraderOrder") or []
        snapshot: list[tuple[str, str]] = []
        for row in trader_orders:
            row_id = str(getattr(row, "id", "") or "")
            status = str(getattr(row, "status", "") or "")
            snapshot.append((row_id, status))
        self.commit_snapshots.append(snapshot)

    @property
    def pre_submit_commit_count(self) -> int:
        """Number of commits that contained at least one row in
        'placing' status — i.e. ``_commit_pre_submit_projection`` ran."""
        return sum(
            1
            for snapshot in self.commit_snapshots
            if any(status == "placing" for _id, status in snapshot)
        )


def _leg_result(
    *,
    leg_id: str,
    status: str = "executed",
    notional_usd: float = 10.0,
    shares: float = 20.0,
    effective_price: float = 0.5,
    payload: dict | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        leg_id=leg_id,
        status=status,
        effective_price=effective_price,
        error_message=None,
        payload=payload
        if payload is not None
        else {
            "provider": "test",
            "token_id": "token-test",
            "filled_size": shares,
            "average_fill_price": effective_price,
            "filled_notional_usd": notional_usd,
        },
        provider_order_id=f"provider-{leg_id}",
        provider_clob_order_id=f"clob-{leg_id}",
        shares=shares,
        notional_usd=notional_usd,
    )


def _common_monkeypatches(monkeypatch, *, engine, legs, submit_results):
    """Wire up the deterministic monkeypatches every test needs."""
    constraints = {"max_unhedged_notional_usd": 0.0, "hedge_timeout_seconds": 20}
    plan = {"policy": "SINGLE_LEG", "plan_id": "plan-preflight", "metadata": {}}
    monkeypatch.setattr(engine, "_build_plan", lambda *a, **k: (plan, legs, constraints))
    monkeypatch.setattr(session_engine_module, "supports_reprice", lambda _policy: False)
    monkeypatch.setattr(session_engine_module, "execution_waves", lambda _policy, leg_rows: [leg_rows])
    monkeypatch.setattr(session_engine_module, "requires_pair_lock", lambda *a, **k: False)
    monkeypatch.setattr(session_engine_module, "set_trade_signal_status", AsyncMock(return_value=True))
    monkeypatch.setattr(session_engine_module, "sync_trader_position_inventory", AsyncMock(return_value={}))
    monkeypatch.setattr(session_engine_module.event_bus, "publish", AsyncMock(return_value=None))
    monkeypatch.setattr(engine, "_publish_hot_signal_status", AsyncMock(return_value=None))
    monkeypatch.setattr(
        intent_runtime_module,
        "get_intent_runtime",
        lambda: SimpleNamespace(update_signal_status=AsyncMock()),
    )
    submit_mock = AsyncMock(return_value=submit_results)
    monkeypatch.setattr(session_engine_module, "submit_execution_wave", submit_mock)
    return submit_mock


@pytest.mark.asyncio
async def test_preflight_happy_path_writes_placeholders_and_submits(monkeypatch):
    """All legs pass preflight → placeholders written → submit invoked."""
    db = _RecordingDb()
    engine = session_engine_module.ExecutionSessionEngine(db)

    legs = [
        _leg(leg_id="leg-happy-1", market_id="market-happy", token_id="11111111111111111111"),
        _leg(leg_id="leg-happy-2", market_id="market-happy", token_id="22222222222222222222"),
    ]

    submit_results = [
        _leg_result(leg_id="leg-happy-1", status="executed"),
        _leg_result(leg_id="leg-happy-2", status="executed"),
    ]
    submit_mock = _common_monkeypatches(
        monkeypatch, engine=engine, legs=legs, submit_results=submit_results
    )

    # Preflight book resolution returns an empty (no measurable spread)
    # book.  With max_spread_bps unset in risk_limits the gate is a no-op
    # regardless, so legs pass preflight.
    _no_book = AsyncMock(return_value=(None, [], None, "test_no_book", None))
    monkeypatch.setattr(venue_gates_module, "_resolve_shadow_book_and_tape", _no_book)
    # BUY collateral gate passes.
    monkeypatch.setattr(
        session_engine_module.live_execution_service,
        "check_buy_pre_submit_gate",
        AsyncMock(return_value=(True, None)),
    )

    signal = _signal(signal_id="signal-happy", market_id="market-happy")
    result = await engine.execute_signal(
        trader_id="trader-happy",
        signal=signal,
        decision_id="decision-happy",
        strategy_key="generic_strategy",
        strategy_version=None,
        strategy_params={},
        risk_limits={},
        mode="live",
        size_usd=20.0,
        reason="happy-path",
    )

    # The submit_execution_wave should have received BOTH legs (none
    # filtered out by preflight).
    assert submit_mock.await_count == 1
    submitted_legs = submit_mock.await_args.kwargs["legs_with_notionals"]
    assert {leg["leg_id"] for leg, _notional in submitted_legs} == {"leg-happy-1", "leg-happy-2"}
    # Placeholders for both legs were committed in the pre-submit
    # projection (status='placing' snapshot).
    assert db.pre_submit_commit_count == 1
    placeholder_snapshot = next(
        snapshot
        for snapshot in db.commit_snapshots
        if any(status == "placing" for _id, status in snapshot)
    )
    placing_rows = [row_id for row_id, status in placeholder_snapshot if status == "placing"]
    assert len(placing_rows) == 2
    # No preflight rejection telemetry was recorded.
    timing = (result.payload or {}).get("execution_timing_ms") or {}
    assert "preflight_legs_pre_rejected" not in timing
    assert "preflight_dbwrites_avoided" not in timing


@pytest.mark.asyncio
async def test_preflight_all_spread_reject_skips_projection_commit(monkeypatch):
    """Every leg fails the spread gate → no projection commit, all skipped."""
    db = _RecordingDb()
    engine = session_engine_module.ExecutionSessionEngine(db)

    legs = [
        _leg(leg_id="leg-allrej-1", market_id="market-allrej", token_id="11111111111111111111"),
        _leg(leg_id="leg-allrej-2", market_id="market-allrej", token_id="22222222222222222222"),
    ]

    # Build a fake book with a wide spread (best_bid=0.20, best_ask=0.80
    # → spread_bps comfortably above 100).
    wide_book = {
        "bids": [{"price": 0.20, "size": 100.0}],
        "asks": [{"price": 0.80, "size": 100.0}],
    }
    _wide = AsyncMock(return_value=(wide_book, [], 0.0, "test_wide_book", None))
    monkeypatch.setattr(venue_gates_module, "_resolve_shadow_book_and_tape", _wide)
    # Force the spread gate to reject deterministically regardless of how
    # _compute_book_spread_bps interprets the book shape.
    def _reject(*, book_payload, risk_limits):
        return (True, 9999.0, 100.0)
    monkeypatch.setattr(session_engine_module, "_check_max_spread_bps", _reject)
    monkeypatch.setattr(venue_gates_module, "_check_max_spread_bps", _reject)
    # BUY gate is irrelevant for this test, but stub it as passing.
    monkeypatch.setattr(
        session_engine_module.live_execution_service,
        "check_buy_pre_submit_gate",
        AsyncMock(return_value=(True, None)),
    )

    submit_mock = _common_monkeypatches(
        monkeypatch, engine=engine, legs=legs, submit_results=[]
    )

    signal = _signal(signal_id="signal-allrej", market_id="market-allrej")
    result = await engine.execute_signal(
        trader_id="trader-allrej",
        signal=signal,
        decision_id="decision-allrej",
        strategy_key="generic_strategy",
        strategy_version=None,
        strategy_params={"max_spread_bps": 100.0},
        risk_limits={"max_spread_bps": 100.0},
        mode="live",
        size_usd=20.0,
        reason="all-reject",
    )

    # submit_execution_wave must NOT have been called — every leg was
    # filtered out by preflight.
    assert submit_mock.await_count == 0
    # _commit_pre_submit_projection must NOT have committed — no
    # placeholder snapshot at all.  (The post-submit
    # ``_persist_execution_projection`` still commits the final
    # failed/skipped session, so we check the placeholder-stage
    # snapshot specifically.)
    assert db.pre_submit_commit_count == 0
    # Preflight telemetry was recorded with N=2.
    timing = (result.payload or {}).get("execution_timing_ms") or {}
    assert timing.get("preflight_legs_pre_rejected") == 2.0
    assert timing.get("preflight_dbwrites_avoided") == 2.0


@pytest.mark.asyncio
async def test_preflight_mixed_only_passing_leg_persists_placeholder(monkeypatch):
    """One leg passes, two fail → only one placeholder written, only one submitted."""
    db = _RecordingDb()
    engine = session_engine_module.ExecutionSessionEngine(db)

    legs = [
        _leg(leg_id="leg-mixed-pass", market_id="market-mixed", token_id="11111111111111111111"),
        _leg(leg_id="leg-mixed-fail-1", market_id="market-mixed", token_id="22222222222222222222"),
        _leg(leg_id="leg-mixed-fail-2", market_id="market-mixed", token_id="33333333333333333333"),
    ]

    # Spread-gate rejects only the two "fail" tokens.  We discriminate by
    # peeking at the most-recent ``_resolve_shadow_book_and_tape`` token_id
    # via a closure — simpler than threading state through _check_max_spread_bps.
    last_token: dict[str, str | None] = {"token_id": None}

    async def _fake_resolve(*, token_id, live_context):
        last_token["token_id"] = token_id
        return ({"bids": [], "asks": []}, [], 0.0, "test_book", None)

    monkeypatch.setattr(venue_gates_module, "_resolve_shadow_book_and_tape", _fake_resolve)

    def _fake_spread(*, book_payload, risk_limits):
        token_id = last_token.get("token_id") or ""
        if token_id in {"22222222222222222222", "33333333333333333333"}:
            return (True, 9999.0, 100.0)
        return (False, None, 100.0)

    monkeypatch.setattr(session_engine_module, "_check_max_spread_bps", _fake_spread)
    monkeypatch.setattr(venue_gates_module, "_check_max_spread_bps", _fake_spread)
    monkeypatch.setattr(
        session_engine_module.live_execution_service,
        "check_buy_pre_submit_gate",
        AsyncMock(return_value=(True, None)),
    )

    submit_results = [_leg_result(leg_id="leg-mixed-pass", status="executed")]
    submit_mock = _common_monkeypatches(
        monkeypatch, engine=engine, legs=legs, submit_results=submit_results
    )

    signal = _signal(signal_id="signal-mixed", market_id="market-mixed")
    result = await engine.execute_signal(
        trader_id="trader-mixed",
        signal=signal,
        decision_id="decision-mixed",
        strategy_key="generic_strategy",
        strategy_version=None,
        strategy_params={"max_spread_bps": 100.0},
        risk_limits={"max_spread_bps": 100.0},
        mode="live",
        size_usd=30.0,
        reason="mixed",
    )

    # submit_execution_wave received ONLY the passing leg.
    assert submit_mock.await_count == 1
    submitted_legs = submit_mock.await_args.kwargs["legs_with_notionals"]
    submitted_ids = {leg["leg_id"] for leg, _ in submitted_legs}
    assert submitted_ids == {"leg-mixed-pass"}
    # Exactly one placeholder commit ran and it contained exactly one
    # row in 'placing' status (the only passing leg).
    assert db.pre_submit_commit_count == 1
    placeholder_snapshot = next(
        snapshot
        for snapshot in db.commit_snapshots
        if any(status == "placing" for _id, status in snapshot)
    )
    placing_rows = [row_id for row_id, status in placeholder_snapshot if status == "placing"]
    assert len(placing_rows) == 1
    # Telemetry: 2 legs pre-rejected, 2 db writes avoided.
    timing = (result.payload or {}).get("execution_timing_ms") or {}
    assert timing.get("preflight_legs_pre_rejected") == 2.0
    assert timing.get("preflight_dbwrites_avoided") == 2.0


@pytest.mark.asyncio
async def test_preflight_buy_collateral_fail_marks_leg_skipped(monkeypatch):
    """BUY collateral gate fails → leg is skipped with reason='buy_pre_submit_gate'."""
    db = _RecordingDb()
    engine = session_engine_module.ExecutionSessionEngine(db)

    legs = [
        _leg(leg_id="leg-buyfail", market_id="market-buyfail", token_id="11111111111111111111"),
    ]

    # Spread gate passes.
    _no_book = AsyncMock(return_value=(None, [], None, "test_no_book", None))
    monkeypatch.setattr(venue_gates_module, "_resolve_shadow_book_and_tape", _no_book)
    def _pass(*, book_payload, risk_limits):
        return (False, None, None)
    monkeypatch.setattr(session_engine_module, "_check_max_spread_bps", _pass)
    monkeypatch.setattr(venue_gates_module, "_check_max_spread_bps", _pass)
    # BUY collateral gate REJECTS.
    monkeypatch.setattr(
        session_engine_module.live_execution_service,
        "check_buy_pre_submit_gate",
        AsyncMock(return_value=(False, "Insufficient collateral: need 10.00, have 0.00.")),
    )

    submit_mock = _common_monkeypatches(
        monkeypatch, engine=engine, legs=legs, submit_results=[]
    )

    signal = _signal(signal_id="signal-buyfail", market_id="market-buyfail")
    result = await engine.execute_signal(
        trader_id="trader-buyfail",
        signal=signal,
        decision_id="decision-buyfail",
        strategy_key="generic_strategy",
        strategy_version=None,
        strategy_params={},
        risk_limits={},
        mode="live",
        size_usd=10.0,
        reason="buy-collateral-fail",
    )

    # submit_execution_wave never called — the only leg was pre-rejected.
    assert submit_mock.await_count == 0
    # _commit_pre_submit_projection did NOT run — no placeholder commit.
    assert db.pre_submit_commit_count == 0
    # The synthesized LegSubmitResult propagated as status='skipped' in
    # the leg_execution_records list (which becomes result.payload['legs']).
    leg_records = (result.payload or {}).get("legs") or []
    assert len(leg_records) == 1
    assert leg_records[0]["status"] == "skipped"
    # The ExecutionSessionLeg row picked up the buy_pre_submit_gate
    # error_message as its last_error.
    leg_rows = db.persisted_rows_by_type.get("ExecutionSessionLeg") or []
    assert len(leg_rows) == 1
    last_error = str(getattr(leg_rows[0], "last_error", "") or "")
    assert "collateral" in last_error.lower() or "buy" in last_error.lower()
    # SessionExecutionResult-level error_message carries the buy gate
    # rejection string (it gets joined with other skip reasons).
    assert "collateral" in str(result.error_message or "").lower()
    # Telemetry recorded.
    timing = (result.payload or {}).get("execution_timing_ms") or {}
    assert timing.get("preflight_legs_pre_rejected") == 1.0
    assert timing.get("preflight_dbwrites_avoided") == 1.0
