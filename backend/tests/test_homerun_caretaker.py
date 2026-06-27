from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "monitoring" / "homerun_caretaker.py"
SPEC = importlib.util.spec_from_file_location("homerun_caretaker", MODULE_PATH)
assert SPEC is not None
caretaker = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = caretaker
SPEC.loader.exec_module(caretaker)


def _state(*, daily_pnl: float = 1.0, gross_exposure: float = 50.0, open_orders: int = 1, stale: bool = False):
    return {
        "health": {
            "ok": True,
            "data": {"checks": {"database": True}},
        },
        "orchestrator": {
            "ok": True,
            "data": {
                "control": {"mode": "live", "kill_switch": False},
                "snapshot": {
                    "daily_pnl": daily_pnl,
                    "gross_exposure_usd": gross_exposure,
                    "open_orders": open_orders,
                    "is_stale": stale,
                    "heartbeat_lag_seconds": 1.0 if not stale else 500.0,
                    "last_error": None,
                },
                "runtime_state": {
                    "worker_stale": stale,
                    "heartbeat_lag_seconds": 1.0 if not stale else 500.0,
                },
            },
        },
        "open_orders": {"ok": True, "items": [{} for _ in range(open_orders)]},
        "traders": {"ok": True, "items": []},
    }


def _policy(**risk_overrides):
    return caretaker.deep_merge(
        caretaker.DEFAULT_POLICY,
        {
            "risk": {
                "max_daily_loss_usd": 10.0,
                "max_gross_exposure_usd": 100.0,
                "max_open_orders": 3,
                "max_heartbeat_lag_seconds": 120.0,
                **risk_overrides,
            }
        },
    )


def test_evaluate_risk_passes_inside_limits():
    breaches = caretaker.evaluate_risk(_state(), _policy())

    assert breaches == []


def test_evaluate_risk_flags_loss_exposure_orders_and_staleness():
    breaches = caretaker.evaluate_risk(
        _state(daily_pnl=-11.0, gross_exposure=150.0, open_orders=4, stale=True),
        _policy(),
    )

    codes = {breach.code for breach in breaches}
    assert codes == {
        "daily_loss_limit",
        "gross_exposure_limit",
        "open_order_limit",
        "orchestrator_stale",
    }
    severities = {breach.code: breach.severity for breach in breaches}
    assert severities["daily_loss_limit"] == "critical"
    assert severities["gross_exposure_limit"] == "high"


def test_equity_floor_breaches_when_below_floor():
    state = _state()
    state["live_balance"] = {"ok": True, "data": {"available": 18.0, "positions_value": 5.0}}
    breaches = caretaker.evaluate_risk(state, _policy(min_account_equity_usd=30.0))

    codes = {breach.code for breach in breaches}
    assert "account_equity_floor" in codes
    floor_breach = next(b for b in breaches if b.code == "account_equity_floor")
    assert floor_breach.severity == "critical"
    assert floor_breach.observed == 23.0


def test_equity_floor_clean_when_above_floor():
    state = _state()
    state["live_balance"] = {"ok": True, "data": {"available": 58.0, "positions_value": 20.0}}
    breaches = caretaker.evaluate_risk(state, _policy(min_account_equity_usd=30.0))

    assert [b.code for b in breaches] == []


def test_equity_floor_skipped_without_balance_snapshot():
    breaches = caretaker.evaluate_risk(_state(), _policy(min_account_equity_usd=30.0))

    assert "account_equity_floor" not in {b.code for b in breaches}
    assert "account_balance_unreadable" not in {b.code for b in breaches}


def test_build_backtest_windows_uses_fixed_policy_windows():
    policy = caretaker.deep_merge(
        caretaker.DEFAULT_POLICY,
        {
            "research": {
                "windows_utc": [
                    {"start": "2026-01-01T00:00:00Z", "end": "2026-01-01T01:00:00Z"},
                    {"start": "", "end": "2026-01-01T02:00:00Z"},
                ]
            }
        },
    )

    assert caretaker.build_backtest_windows(policy) == [
        {"start": "2026-01-01T00:00:00Z", "end": "2026-01-01T01:00:00Z"}
    ]
