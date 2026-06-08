"""A.3 safety gate — ``reconcile_live_positions(place_exits=False)`` must never
mutate the venue.

The cold *reconciliation* plane runs ``reconcile_live_positions`` in
detection-only mode (``place_exits=False``): it computes every close decision
for telemetry but must place / cancel **zero** live orders.  The whole money
path stays single-executor on the trading plane.

The 4 execution primitives reached from inside ``reconcile_live_positions`` are::

    live_execution_service.cancel_order(...)     # cancel a working order
    execute_live_order(...)                       # place a sell
    exit_executor.run_exit_pass(...)             # laddered submit/escalate/reprice
    _prepare_sell_allowance_bounded(...)         # on-chain allowance top-up

Every one of those calls is funnelled through a ``_gx_*`` wrapper closure that
short-circuits when ``place_exits`` is False (the two ladder cancel callbacks
instead guard inline with ``place_exits and ...``).  This is the *single choke
point*.

This test enforces that structurally with an AST scan: a brand-new execution
call added to the function in the future — without routing through a wrapper or
a ``place_exits`` guard — fails here, *before* it can let the detection plane
double-execute against the live trader.  It needs no DB and covers every code
path, not just the ones a fixture happens to exercise.
"""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

_LIFECYCLE = (
    Path(__file__).resolve().parents[1]
    / "services"
    / "trader_orchestrator"
    / "position_lifecycle.py"
)

# Bare attribute / function names that mutate the venue (or its allowance).
_GATED_PRIMITIVES = {
    "cancel_order",
    "execute_live_order",
    "run_exit_pass",
    "_prepare_sell_allowance_bounded",
}


def _reconcile_fn(tree: ast.AST) -> ast.AsyncFunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "reconcile_live_positions":
            return node
    raise AssertionError("reconcile_live_positions not found in position_lifecycle.py")


def _call_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def test_place_exits_is_single_execution_choke_point():
    """Every venue-mutating call in reconcile_live_positions is gated by
    ``place_exits`` — via a ``_gx_*`` wrapper or an inline guard."""
    src = _LIFECYCLE.read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = _reconcile_fn(tree)
    src_lines = src.splitlines()

    # Line spans of the ``_gx_*`` wrapper closures.  A primitive call inside one
    # of these IS the gate (each wrapper is ``if not place_exits: return <noop>``
    # then the real call), so those are expected and allowed.
    wrapper_spans = [
        (node.lineno, node.end_lineno)
        for node in ast.walk(fn)
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef))
        and node.name.startswith("_gx_")
    ]
    assert len(wrapper_spans) == 4, (
        f"expected 4 _gx_* execution wrappers in reconcile_live_positions, "
        f"found {len(wrapper_spans)} — the detection-only gate is incomplete"
    )

    def _inside_wrapper(lineno: int) -> bool:
        return any(lo <= lineno <= (hi or lo) for lo, hi in wrapper_spans)

    offenders: list[tuple[int, str, str]] = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node)
        if name not in _GATED_PRIMITIVES:
            continue
        lineno = node.lineno
        if _inside_wrapper(lineno):
            continue  # the wrapper itself — the gate
        line_text = src_lines[lineno - 1]
        if "place_exits" in line_text:
            continue  # inline-guarded callback (``place_exits and await ...``)
        offenders.append((lineno, name, line_text.strip()))

    assert not offenders, (
        "Ungated venue-mutating call(s) in reconcile_live_positions — in "
        "detection-only mode (place_exits=False) these would let the cold "
        "reconciliation plane place/cancel live orders and double-execute "
        "against the trading plane. Route each through a _gx_* wrapper or guard "
        "it with `place_exits`:\n"
        + "\n".join(f"  line {ln}: {nm}() -> {txt}" for ln, nm, txt in offenders)
    )


def test_gx_wrappers_present_and_short_circuit_on_place_exits():
    """Each ``_gx_*`` wrapper must early-return when ``place_exits`` is False
    (i.e. the gate exists, not just the call-site renames)."""
    src = _LIFECYCLE.read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = _reconcile_fn(tree)

    wrappers = {
        node.name: node
        for node in ast.walk(fn)
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef))
        and node.name.startswith("_gx_")
    }
    expected = {
        "_gx_cancel_order",
        "_gx_prepare_sell_allowance",
        "_gx_execute_live_order",
        "_gx_run_exit_pass",
    }
    assert expected.issubset(wrappers), (
        f"missing execution wrappers: {sorted(expected - set(wrappers))}"
    )

    for wname, wnode in wrappers.items():
        # First statement must be ``if not place_exits: return ...`` so the
        # primitive below is unreachable in detection-only mode.
        guard = wnode.body[0]
        assert isinstance(guard, ast.If), f"{wname}: first statement is not an `if` guard"
        cond = ast.unparse(guard.test).replace(" ", "")
        assert cond == "notplace_exits", (
            f"{wname}: guard is `{ast.unparse(guard.test)}`, expected `not place_exits`"
        )
        assert any(isinstance(s, ast.Return) for s in guard.body), (
            f"{wname}: place_exits guard does not return before the primitive call"
        )


@pytest.mark.asyncio
async def test_detection_only_reconcile_places_no_orders(tmp_path, monkeypatch):
    """Runtime proof of the A.3 invariant over a real reconcile pass.

    ``test_live_exit_submission_uses_ioc_for_rapid_strategy_close`` shows this
    exact fixture (a live position whose strategy says "close") drives
    ``execute_live_order`` exactly once on the trading plane.  Here the SAME
    fixture under the cold reconciliation plane (``place_exits=False``) must
    place / cancel ZERO live orders — and the detection path must still run to
    completion (exercises the wrappers' no-op return shapes through the
    downstream bookkeeping, catching attribute-shape regressions).
    """
    from models.database import Base, TraderOrder  # noqa: F401
    from services.strategies.base import ExitDecision
    from services.trader_orchestrator import position_lifecycle
    from tests.postgres_test_db import build_postgres_session_factory
    from tests.test_trader_position_lifecycle_resolution import _seed_order

    engine, session_factory = await build_postgres_session_factory(
        Base, "reconcile_place_exits_gate"
    )
    try:
        async with session_factory() as session:
            await _seed_order(
                session,
                mode="live",
                status="open",
                payload_json={
                    "strategy_type": "btc_eth_maker_quote",
                    "token_id": "token-1",
                    "provider_reconciliation": {
                        "filled_size": 5.0,
                        "average_fill_price": 0.4,
                        "filled_notional_usd": 2.0,
                    },
                },
            )
            monkeypatch.setattr(
                position_lifecycle,
                "load_market_info_for_orders",
                AsyncMock(
                    return_value={
                        "market-1": {
                            "closed": False,
                            "accepting_orders": True,
                            "winner": None,
                            "winning_outcome": None,
                            "outcome_prices": [0.5, 0.5],
                        }
                    }
                ),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_positions_by_token",
                AsyncMock(return_value={"token-1": {"asset": "token-1", "size": 5.0, "curPrice": 0.5}}),
            )
            monkeypatch.setattr(
                position_lifecycle,
                "_load_execution_wallet_recent_sell_trades_by_token",
                AsyncMock(return_value={}),
            )

            class _RapidStrategy:
                def should_exit(self, _position, _market_state):
                    return ExitDecision(
                        action="close",
                        reason="Rapid hard-cap take profit (23.8% >= 18.0%)",
                    )

            monkeypatch.setattr(
                "services.strategy_loader.strategy_loader.get_strategy",
                lambda _slug: SimpleNamespace(instance=_RapidStrategy()),
            )

            # The four venue-mutating primitives — every one must stay untouched.
            execute_mock = AsyncMock(
                return_value=SimpleNamespace(
                    status="failed",
                    error_message="should_not_be_called",
                    order_id=None,
                    payload={},
                )
            )
            cancel_mock = AsyncMock(return_value=True)
            run_exit_pass_mock = AsyncMock(return_value={})
            prepare_mock = AsyncMock(return_value=True)
            monkeypatch.setattr("services.live_execution_adapter.execute_live_order", execute_mock)
            monkeypatch.setattr(position_lifecycle.live_execution_service, "cancel_order", cancel_mock)
            monkeypatch.setattr(position_lifecycle.exit_executor, "run_exit_pass", run_exit_pass_mock)
            monkeypatch.setattr(position_lifecycle, "_prepare_sell_allowance_bounded", prepare_mock)

            result = await position_lifecycle.reconcile_live_positions(
                session,
                trader_id="trader-1",
                trader_params={},
                dry_run=False,
                place_exits=False,
            )

            assert execute_mock.await_count == 0, "execute_live_order called in detection-only mode"
            assert cancel_mock.await_count == 0, "cancel_order called in detection-only mode"
            assert run_exit_pass_mock.await_count == 0, "run_exit_pass called in detection-only mode"
            assert prepare_mock.await_count == 0, "prepare_sell_allowance called in detection-only mode"
            # Detection still completed (no crash on the wrappers' no-op returns).
            assert isinstance(result, dict)
    finally:
        await engine.dispose()
