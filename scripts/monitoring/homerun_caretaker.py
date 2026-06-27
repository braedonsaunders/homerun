from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT_DIR = REPO_ROOT / "output" / "caretaker"
DEFAULT_API_BASE_URL = os.environ.get("HOMERUN_API_BASE_URL", "http://127.0.0.1:8000")
TERMINAL_BACKTEST_STATUSES = {"completed", "failed", "cancelled", "error"}


DEFAULT_POLICY: dict[str, Any] = {
    "operator": "codex_caretaker",
    "risk": {
        "max_daily_loss_usd": 25.0,
        "max_gross_exposure_usd": 500.0,
        "max_open_orders": 12,
        "max_heartbeat_lag_seconds": 120.0,
        "stop_live_on_breach": True,
        "cancel_live_orders_on_critical_breach": True,
        "block_traders_on_breach": True,
        "kill_switch_on_breach": True,
    },
    "live": {
        "allow_live_start": False,
        "allow_manual_orders": False,
        "allow_trader_start": False,
        "selected_account_id": "live:polymarket",
        "apply_orchestrator_settings": False,
        "apply_trader_overrides": False,
        "unblock_managed_traders": False,
        "run_managed_traders_once": False,
        "managed_trader_ids": [],
        "managed_strategy_slugs": ["tail_end_carry"],
        "global_risk": {},
        "global_runtime_live_risk_clamps": {},
        "managed_strategy_params": {},
        "managed_risk_limits": {},
    },
    "research": {
        "enabled": True,
        "candidate_strategy_slugs": ["tail_end_carry"],
        "include_running_trader_params": True,
        "rolling_window_hours": 3,
        "window_hours": 1,
        "exclude_recent_minutes": 15,
        "max_runs_per_cycle": 3,
        "poll_timeout_seconds": 1800,
        "poll_interval_seconds": 5.0,
        "initial_capital_usd": 1000.0,
        "counterfactual_sample_size": 2,
        "ensemble_sample_size": 2,
        "fills_sample_size": 1000,
    },
    "promotion": {
        "allow_strategy_updates": False,
        "allow_trader_config_updates": False,
        "min_total_return_pct": 0.0,
        "min_trade_count": 5,
        "min_return_improvement_pct": 0.02,
        "require_all_windows_completed": True,
    },
}


@dataclass(frozen=True)
class RiskBreach:
    code: str
    severity: str
    message: str
    observed: float | str | None = None
    limit: float | str | None = None


class HomerunApiError(RuntimeError):
    def __init__(self, method: str, url: str, status: int | None, body: str):
        self.method = method
        self.url = url
        self.status = status
        self.body = body
        status_text = "connection_error" if status is None else str(status)
        super().__init__(f"{method} {url} failed ({status_text}): {body}")


class HomerunApiClient:
    def __init__(self, base_url: str, timeout_seconds: float = 20.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def get(self, path: str, query: dict[str, Any] | None = None) -> Any:
        return self.request("GET", path, query=query)

    def post(self, path: str, payload: dict[str, Any] | None = None) -> Any:
        return self.request("POST", path, payload=payload or {})

    def put(self, path: str, payload: dict[str, Any]) -> Any:
        return self.request("PUT", path, payload=payload)

    def request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        query: dict[str, Any] | None = None,
    ) -> Any:
        url = self._url(path, query=query)
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
                if not body:
                    return {}
                return json.loads(body)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise HomerunApiError(method, url, exc.code, body) from exc
        except urllib.error.URLError as exc:
            raise HomerunApiError(method, url, None, str(exc.reason)) from exc
        except TimeoutError as exc:
            raise HomerunApiError(method, url, None, "request timed out") from exc

    def _url(self, path: str, query: dict[str, Any] | None = None) -> str:
        if path.startswith("http://") or path.startswith("https://"):
            base = path
        else:
            base = f"{self.base_url}/{path.lstrip('/')}"
        if not query:
            return base
        cleaned = {
            key: value
            for key, value in query.items()
            if value is not None
        }
        if not cleaned:
            return base
        return f"{base}?{urllib.parse.urlencode(cleaned)}"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_policy(path: Path | None) -> dict[str, Any]:
    policy = dict(DEFAULT_POLICY)
    if path is None:
        return policy
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Policy file must contain a JSON object: {path}")
    return deep_merge(policy, raw)


def to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_call(label: str, func) -> dict[str, Any]:
    try:
        return {"ok": True, "data": func()}
    except HomerunApiError as exc:
        return {
            "ok": False,
            "error": str(exc),
            "status": exc.status,
            "body": exc.body,
            "label": label,
        }


def summarize_traders(payload: Any) -> list[dict[str, Any]]:
    raw_traders = payload.get("traders") if isinstance(payload, dict) else []
    if not isinstance(raw_traders, list):
        return []
    summarized: list[dict[str, Any]] = []
    for trader in raw_traders:
        if not isinstance(trader, dict):
            continue
        source_configs = []
        for cfg in trader.get("source_configs") or []:
            if not isinstance(cfg, dict):
                continue
            source_configs.append(
                {
                    "source_key": cfg.get("source_key"),
                    "strategy_key": cfg.get("strategy_key"),
                    "strategy_version": cfg.get("strategy_version"),
                    "strategy_params": cfg.get("strategy_params") or {},
                }
            )
        risk_limits = trader.get("risk_limits") if isinstance(trader.get("risk_limits"), dict) else {}
        summarized.append(
            {
                "id": trader.get("id"),
                "name": trader.get("name"),
                "mode": trader.get("mode"),
                "source_configs": source_configs,
                "is_enabled": bool(trader.get("is_enabled")),
                "is_paused": bool(trader.get("is_paused")),
                "block_new_orders": bool(trader.get("block_new_orders")),
                "last_run_at": trader.get("last_run_at"),
                "updated_at": trader.get("updated_at"),
                "risk_limits": {
                    key: risk_limits.get(key)
                    for key in (
                        "max_trade_notional_usd",
                        "max_position_notional_usd",
                        "max_open_orders",
                        "max_open_positions",
                        "max_daily_loss_usd",
                        "max_daily_spend_usd",
                        "max_gross_exposure_usd",
                        "max_orders_per_cycle",
                    )
                    if key in risk_limits
                },
            }
        )
    return summarized


def summarize_orders(payload: Any) -> list[dict[str, Any]]:
    raw_orders = payload.get("orders") if isinstance(payload, dict) else []
    if not isinstance(raw_orders, list):
        return []
    orders: list[dict[str, Any]] = []
    for order in raw_orders:
        if not isinstance(order, dict):
            continue
        question = str(order.get("market_question") or "")
        orders.append(
            {
                "id": order.get("id"),
                "trader_id": order.get("trader_id"),
                "source": order.get("source"),
                "strategy_key": order.get("strategy_key"),
                "strategy_version": order.get("strategy_version"),
                "market_id": order.get("market_id"),
                "market_question": question[:180],
                "direction": order.get("direction"),
                "mode": order.get("mode"),
                "status": order.get("status"),
                "notional_usd": order.get("notional_usd"),
                "filled_notional_usd": order.get("filled_notional_usd"),
                "entry_price": order.get("entry_price"),
                "effective_price": order.get("effective_price"),
                "current_price": order.get("current_price"),
                "unrealized_pnl": order.get("unrealized_pnl"),
                "actual_profit": order.get("actual_profit"),
                "edge_percent": order.get("edge_percent"),
                "created_at": order.get("created_at"),
                "updated_at": order.get("updated_at"),
            }
        )
    return orders


def summarize_strategies(payload: Any) -> list[dict[str, Any]]:
    items = payload.get("items") if isinstance(payload, dict) else []
    if not isinstance(items, list):
        return []
    strategies: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        strategies.append(
            {
                "id": item.get("id"),
                "slug": item.get("slug"),
                "source_key": item.get("source_key"),
                "name": item.get("name"),
                "enabled": bool(item.get("enabled")),
                "status": item.get("status"),
                "version": item.get("version"),
                "config": item.get("config") or {},
                "error_message": item.get("error_message"),
            }
        )
    return strategies


def collect_operational_state(
    api: HomerunApiClient,
    *,
    include_strategies: bool = False,
    deep: bool = False,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "timestamp": iso_utc(utc_now()),
        "api_base_url": api.base_url,
    }
    health = safe_call("health_gui", lambda: api.get("/health/gui"))
    if not health["ok"]:
        health = safe_call("health", lambda: api.get("/health"))
    state["health"] = health
    state["orchestrator"] = safe_call(
        "trader_orchestrator_status",
        lambda: api.get("/api/trader-orchestrator/status"),
    )
    traders = safe_call("traders", lambda: api.get("/api/traders"))
    state["traders"] = {
        "ok": traders["ok"],
        "error": traders.get("error"),
        "items": summarize_traders(traders.get("data")),
    }
    open_orders = safe_call(
        "open_trader_orders",
        lambda: api.get(
            "/api/traders/orders/all",
            {"status": "open", "limit": 500, "since_seconds": 0},
        ),
    )
    state["open_orders"] = {
        "ok": open_orders["ok"],
        "error": open_orders.get("error"),
        "items": summarize_orders(open_orders.get("data")),
    }
    state["backtest_runs"] = safe_call(
        "backtest_runs",
        lambda: api.get("/api/backtest/runs", {"limit": 20}),
    )
    if include_strategies:
        strategies = safe_call(
            "strategies",
            lambda: api.get("/api/strategy-manager", {"enabled": "true"}),
        )
        state["strategies"] = {
            "ok": strategies["ok"],
            "error": strategies.get("error"),
            "items": summarize_strategies(strategies.get("data")),
        }
    if deep:
        state["live_status"] = safe_call(
            "live_status",
            lambda: api.get("/api/trader-orchestrator/live/status"),
        )
        state["live_balance"] = safe_call(
            "live_balance",
            lambda: api.get("/api/trader-orchestrator/live/balance"),
        )
    return state


def evaluate_risk(state: dict[str, Any], policy: dict[str, Any]) -> list[RiskBreach]:
    risk_policy = policy.get("risk") if isinstance(policy.get("risk"), dict) else {}
    breaches: list[RiskBreach] = []

    health_call = state.get("health") if isinstance(state.get("health"), dict) else {}
    health = health_call.get("data") if health_call.get("ok") else {}
    if not health_call.get("ok"):
        breaches.append(
            RiskBreach(
                code="backend_unreachable",
                severity="critical",
                message="Caretaker could not reach Homerun health endpoint.",
                observed=health_call.get("error"),
                limit="reachable_backend",
            )
        )
    elif isinstance(health, dict):
        checks = health.get("checks") if isinstance(health.get("checks"), dict) else {}
        if checks.get("database") is False:
            breaches.append(
                RiskBreach(
                    code="database_unhealthy",
                    severity="critical",
                    message="Database health check is failing.",
                    observed="database=false",
                    limit="database=true",
                )
            )

    orch_call = state.get("orchestrator") if isinstance(state.get("orchestrator"), dict) else {}
    orch = orch_call.get("data") if orch_call.get("ok") else {}
    if not orch_call.get("ok"):
        breaches.append(
            RiskBreach(
                code="orchestrator_status_unreachable",
                severity="critical",
                message="Caretaker could not read trader orchestrator status.",
                observed=orch_call.get("error"),
                limit="reachable_orchestrator",
            )
        )
        return breaches

    if not isinstance(orch, dict):
        return breaches

    snapshot = orch.get("snapshot") if isinstance(orch.get("snapshot"), dict) else {}
    runtime_state = orch.get("runtime_state") if isinstance(orch.get("runtime_state"), dict) else {}

    daily_pnl = to_float(snapshot.get("daily_pnl"))
    max_daily_loss = to_float(risk_policy.get("max_daily_loss_usd"), 25.0)
    if daily_pnl < -max_daily_loss:
        breaches.append(
            RiskBreach(
                code="daily_loss_limit",
                severity="critical",
                message="Daily PnL breached the caretaker loss limit.",
                observed=daily_pnl,
                limit=-max_daily_loss,
            )
        )

    gross_exposure = to_float(snapshot.get("gross_exposure_usd"))
    max_gross_exposure = to_float(risk_policy.get("max_gross_exposure_usd"), 500.0)
    if gross_exposure > max_gross_exposure:
        breaches.append(
            RiskBreach(
                code="gross_exposure_limit",
                severity="high",
                message="Gross exposure is above the caretaker exposure limit.",
                observed=gross_exposure,
                limit=max_gross_exposure,
            )
        )

    snapshot_open_orders = snapshot.get("open_orders")
    if snapshot_open_orders is None:
        open_orders = len((state.get("open_orders") or {}).get("items") or [])
    else:
        open_orders = to_int(snapshot_open_orders)
    max_open_orders = to_int(risk_policy.get("max_open_orders"), 12)
    if open_orders > max_open_orders:
        breaches.append(
            RiskBreach(
                code="open_order_limit",
                severity="high",
                message="Open order count is above the caretaker limit.",
                observed=open_orders,
                limit=max_open_orders,
            )
        )

    heartbeat_lag = to_float(
        runtime_state.get("heartbeat_lag_seconds"),
        to_float(snapshot.get("heartbeat_lag_seconds")),
    )
    max_lag = to_float(risk_policy.get("max_heartbeat_lag_seconds"), 120.0)
    if bool(runtime_state.get("worker_stale")) or bool(snapshot.get("is_stale")) or heartbeat_lag > max_lag:
        breaches.append(
            RiskBreach(
                code="orchestrator_stale",
                severity="critical",
                message="Trader orchestrator heartbeat is stale.",
                observed=heartbeat_lag,
                limit=max_lag,
            )
        )

    last_error = snapshot.get("last_error")
    if last_error:
        breaches.append(
            RiskBreach(
                code="orchestrator_error",
                severity="high",
                message="Trader orchestrator reported an error.",
                observed=str(last_error)[:240],
                limit="no_last_error",
            )
        )

    # Catastrophe stop: hard floor on total account equity (cash + open positions).
    # This is the "do not let the account go to zero" backstop. When breached it is
    # critical, so enforce_risk_breaches will kill-switch, emergency-stop live, stop
    # live, and block every trader. Only evaluated when a balance snapshot is present
    # (guard/cycle fetch it; plain status without --deep skips it).
    min_equity = risk_policy.get("min_account_equity_usd")
    if min_equity is not None and "live_balance" in state:
        balance_call = state.get("live_balance") if isinstance(state.get("live_balance"), dict) else {}
        if balance_call.get("ok"):
            bal = balance_call.get("data") if isinstance(balance_call.get("data"), dict) else {}
            equity = to_float(bal.get("available")) + to_float(bal.get("positions_value"))
            floor = to_float(min_equity)
            if equity < floor:
                breaches.append(
                    RiskBreach(
                        code="account_equity_floor",
                        severity="critical",
                        message="Account equity fell below the caretaker catastrophe floor.",
                        observed=equity,
                        limit=floor,
                    )
                )
        else:
            breaches.append(
                RiskBreach(
                    code="account_balance_unreadable",
                    severity="high",
                    message="Caretaker could not read live balance to enforce the equity floor.",
                    observed=balance_call.get("error"),
                    limit="readable_balance",
                )
            )

    return breaches


def enforce_risk_breaches(
    api: HomerunApiClient,
    state: dict[str, Any],
    breaches: list[RiskBreach],
    policy: dict[str, Any],
    *,
    dry_run: bool,
) -> list[dict[str, Any]]:
    if not breaches:
        return []
    risk_policy = policy.get("risk") if isinstance(policy.get("risk"), dict) else {}
    operator = str(policy.get("operator") or "codex_caretaker")
    actions: list[dict[str, Any]] = []
    critical = any(breach.severity == "critical" for breach in breaches)

    def record(action: str, result: Any, ok: bool = True) -> None:
        actions.append({"action": action, "ok": ok, "result": result})

    def call(action: str, func) -> None:
        if dry_run:
            record(action, {"dry_run": True})
            return
        result = safe_call(action, func)
        record(action, result.get("data") if result.get("ok") else result, ok=bool(result.get("ok")))

    if bool(risk_policy.get("kill_switch_on_breach", True)):
        call(
            "trader_orchestrator_kill_switch",
            lambda: api.post(
                "/api/trader-orchestrator/kill-switch",
                {"enabled": True, "requested_by": operator},
            ),
        )

    orch = ((state.get("orchestrator") or {}).get("data") or {})
    control = orch.get("control") if isinstance(orch, dict) and isinstance(orch.get("control"), dict) else {}
    if str(control.get("mode") or "").lower() == "live":
        if critical and bool(risk_policy.get("cancel_live_orders_on_critical_breach", True)):
            call("live_emergency_stop", lambda: api.post("/api/trader-orchestrator/live/emergency-stop", {}))
        if bool(risk_policy.get("stop_live_on_breach", True)):
            call(
                "trader_orchestrator_live_stop",
                lambda: api.post(
                    "/api/trader-orchestrator/live/stop",
                    {"requested_by": operator},
                ),
            )

    if bool(risk_policy.get("block_traders_on_breach", True)):
        traders = ((state.get("traders") or {}).get("items") or [])
        for trader in traders:
            if not isinstance(trader, dict):
                continue
            trader_id = str(trader.get("id") or "")
            if not trader_id or bool(trader.get("block_new_orders")):
                continue
            if not bool(trader.get("is_enabled")):
                continue
            call(
                f"block_new_orders:{trader_id}",
                lambda trader_id=trader_id: api.post(
                    f"/api/traders/{trader_id}/block-new-orders",
                    {
                        "enabled": True,
                        "requested_by": operator,
                        "reason": "caretaker risk breach",
                    },
                ),
            )

    return actions


def maintain_live_runtime(
    api: HomerunApiClient,
    state: dict[str, Any],
    policy: dict[str, Any],
    *,
    dry_run: bool,
) -> list[dict[str, Any]]:
    live_policy = policy.get("live") if isinstance(policy.get("live"), dict) else {}
    operator = str(policy.get("operator") or "codex_caretaker")
    actions: list[dict[str, Any]] = []

    def record(action: str, result: Any, ok: bool = True) -> None:
        actions.append({"action": action, "ok": ok, "result": result})

    def call(action: str, func) -> Any:
        if dry_run:
            result = {"dry_run": True}
            record(action, result)
            return result
        result = safe_call(action, func)
        record(action, result.get("data") if result.get("ok") else result, ok=bool(result.get("ok")))
        return result

    orch = ((state.get("orchestrator") or {}).get("data") or {})
    control = orch.get("control") if isinstance(orch, dict) and isinstance(orch.get("control"), dict) else {}
    mode = str(control.get("mode") or "").strip().lower()
    enabled = bool(control.get("is_enabled"))
    paused = bool(control.get("is_paused"))

    if bool(live_policy.get("apply_orchestrator_settings", False)):
        settings_payload = build_orchestrator_settings_payload(policy)
        if settings_payload:
            call(
                "apply_live_orchestrator_settings",
                lambda settings_payload=settings_payload: api.put(
                    "/api/trader-orchestrator/settings",
                    {**settings_payload, "requested_by": operator},
                ),
            )

    if mode != "live" or not enabled or paused:
        if bool(live_policy.get("allow_live_start", False)):
            selected_account_id = str(live_policy.get("selected_account_id") or "live:polymarket")
            start_result = start_live_orchestrator(api, selected_account_id, operator, dry_run=dry_run)
            actions.extend(start_result)
        else:
            record(
                "live_start_skipped",
                {
                    "reason": "live start disabled by policy",
                    "mode": mode,
                    "enabled": enabled,
                    "paused": paused,
                },
            )

    managed_actions = maintain_managed_live_traders(api, state, policy, dry_run=dry_run)
    actions.extend(managed_actions)
    return actions


def build_orchestrator_settings_payload(policy: dict[str, Any]) -> dict[str, Any]:
    live_policy = policy.get("live") if isinstance(policy.get("live"), dict) else {}
    payload: dict[str, Any] = {}
    global_risk = live_policy.get("global_risk")
    if isinstance(global_risk, dict) and global_risk:
        payload["global_risk"] = {
            "max_gross_exposure_usd": to_float(global_risk.get("max_gross_exposure_usd"), 300.0),
            "max_daily_loss_usd": to_float(global_risk.get("max_daily_loss_usd"), 10.0),
            "max_orders_per_cycle": to_int(global_risk.get("max_orders_per_cycle"), 3),
        }
    clamps = live_policy.get("global_runtime_live_risk_clamps")
    if isinstance(clamps, dict) and clamps:
        payload["global_runtime"] = {
            "live_risk_clamps": clamps,
        }
    return payload


def start_live_orchestrator(
    api: HomerunApiClient,
    selected_account_id: str,
    operator: str,
    *,
    dry_run: bool,
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []

    def record(action: str, result: Any, ok: bool = True) -> None:
        actions.append({"action": action, "ok": ok, "result": result})

    if dry_run:
        record(
            "live_start_sequence",
            {"dry_run": True, "selected_account_id": selected_account_id},
        )
        return actions

    preflight = safe_call(
        "live_preflight",
        lambda: api.post(
            "/api/trader-orchestrator/live/preflight",
            {"mode": "live", "requested_by": operator},
        ),
    )
    record("live_preflight", preflight.get("data") if preflight.get("ok") else preflight, ok=bool(preflight.get("ok")))
    if not preflight.get("ok"):
        return actions
    preflight_data = preflight.get("data") if isinstance(preflight.get("data"), dict) else {}
    if preflight_data.get("status") != "passed":
        record("live_start_skipped", {"reason": "preflight did not pass", "preflight": preflight_data}, ok=False)
        return actions

    preflight_id = str(preflight_data.get("preflight_id") or "")
    arm = safe_call(
        "live_arm",
        lambda: api.post(
            "/api/trader-orchestrator/live/arm",
            {"preflight_id": preflight_id, "ttl_seconds": 300, "requested_by": operator},
        ),
    )
    record("live_arm", arm.get("data") if arm.get("ok") else arm, ok=bool(arm.get("ok")))
    if not arm.get("ok"):
        return actions
    arm_data = arm.get("data") if isinstance(arm.get("data"), dict) else {}
    arm_token = str(arm_data.get("arm_token") or "")
    if not arm_token:
        record("live_start_skipped", {"reason": "arm response did not include arm_token"}, ok=False)
        return actions

    started = safe_call(
        "live_start",
        lambda: api.post(
            "/api/trader-orchestrator/live/start",
            {
                "arm_token": arm_token,
                "mode": "live",
                "selected_account_id": selected_account_id,
                "requested_by": operator,
            },
        ),
    )
    record("live_start", started.get("data") if started.get("ok") else started, ok=bool(started.get("ok")))
    return actions


def maintain_managed_live_traders(
    api: HomerunApiClient,
    state: dict[str, Any],
    policy: dict[str, Any],
    *,
    dry_run: bool,
) -> list[dict[str, Any]]:
    live_policy = policy.get("live") if isinstance(policy.get("live"), dict) else {}
    operator = str(policy.get("operator") or "codex_caretaker")
    managed_slugs = {
        str(item).strip()
        for item in (live_policy.get("managed_strategy_slugs") or [])
        if str(item).strip()
    }
    managed_trader_ids = {
        str(item).strip()
        for item in (live_policy.get("managed_trader_ids") or [])
        if str(item).strip()
    }
    if not managed_slugs and not managed_trader_ids:
        return []
    traders = ((state.get("traders") or {}).get("items") or [])
    actions: list[dict[str, Any]] = []
    for trader in traders:
        if not isinstance(trader, dict):
            continue
        if str(trader.get("mode") or "").strip().lower() != "live":
            continue
        trader_slugs = {
            str(cfg.get("strategy_key") or "").strip()
            for cfg in (trader.get("source_configs") or [])
            if isinstance(cfg, dict)
        }
        if not trader_slugs.intersection(managed_slugs):
            continue
        trader_id = str(trader.get("id") or "")
        if not trader_id:
            continue
        if managed_trader_ids and trader_id not in managed_trader_ids:
            continue

        if bool(live_policy.get("apply_trader_overrides", False)):
            actions.append(
                apply_managed_trader_overrides(api, trader_id, managed_slugs, policy, dry_run=dry_run)
            )

        if bool(live_policy.get("unblock_managed_traders", False)) and bool(trader.get("block_new_orders")):
            actions.append(
                call_or_dry_run(
                    api,
                    dry_run,
                    f"unblock_new_orders:{trader_id}",
                    lambda trader_id=trader_id: api.post(
                        f"/api/traders/{trader_id}/block-new-orders",
                        {
                            "enabled": False,
                            "requested_by": operator,
                            "reason": "caretaker live managed trader",
                        },
                    ),
                )
            )

        if bool(live_policy.get("allow_trader_start", False)) and bool(trader.get("is_paused")):
            actions.append(
                call_or_dry_run(
                    api,
                    dry_run,
                    f"start_trader:{trader_id}",
                    lambda trader_id=trader_id: api.post(
                        f"/api/traders/{trader_id}/start",
                        {"requested_by": operator, "copy_existing_positions": False},
                    ),
                )
            )

        if bool(live_policy.get("run_managed_traders_once", False)) and not bool(trader.get("is_paused")):
            actions.append(
                call_or_dry_run(
                    api,
                    dry_run,
                    f"run_once:{trader_id}",
                    lambda trader_id=trader_id: api.post(f"/api/traders/{trader_id}/run-once", {}),
                )
            )
    return actions


def call_or_dry_run(api: HomerunApiClient, dry_run: bool, action: str, func) -> dict[str, Any]:
    del api
    if dry_run:
        return {"action": action, "ok": True, "result": {"dry_run": True}}
    result = safe_call(action, func)
    return {
        "action": action,
        "ok": bool(result.get("ok")),
        "result": result.get("data") if result.get("ok") else result,
    }


def apply_managed_trader_overrides(
    api: HomerunApiClient,
    trader_id: str,
    managed_slugs: set[str],
    policy: dict[str, Any],
    *,
    dry_run: bool,
) -> dict[str, Any]:
    live_policy = policy.get("live") if isinstance(policy.get("live"), dict) else {}
    operator = str(policy.get("operator") or "codex_caretaker")
    strategy_param_overrides = (
        live_policy.get("managed_strategy_params")
        if isinstance(live_policy.get("managed_strategy_params"), dict)
        else {}
    )
    risk_limit_overrides = (
        live_policy.get("managed_risk_limits")
        if isinstance(live_policy.get("managed_risk_limits"), dict)
        else {}
    )

    if dry_run:
        return {
            "action": f"apply_trader_overrides:{trader_id}",
            "ok": True,
            "result": {"dry_run": True},
        }

    trader_result = safe_call(f"get_trader:{trader_id}", lambda: api.get(f"/api/traders/{trader_id}"))
    if not trader_result.get("ok"):
        return {
            "action": f"apply_trader_overrides:{trader_id}",
            "ok": False,
            "result": trader_result,
        }
    trader = trader_result.get("data")
    if not isinstance(trader, dict):
        return {
            "action": f"apply_trader_overrides:{trader_id}",
            "ok": False,
            "result": "trader payload was not an object",
        }

    changed = False
    source_configs = []
    for cfg in trader.get("source_configs") or []:
        if not isinstance(cfg, dict):
            continue
        next_cfg = dict(cfg)
        slug = str(next_cfg.get("strategy_key") or "").strip()
        params_override = strategy_param_overrides.get(slug)
        if slug in managed_slugs and isinstance(params_override, dict) and params_override:
            current_params = next_cfg.get("strategy_params") if isinstance(next_cfg.get("strategy_params"), dict) else {}
            next_params = deep_merge(current_params, params_override)
            if next_params != current_params:
                next_cfg["strategy_params"] = next_params
                changed = True
        source_configs.append(next_cfg)

    risk_limits = trader.get("risk_limits") if isinstance(trader.get("risk_limits"), dict) else {}
    next_risk_limits = dict(risk_limits)
    for slug in managed_slugs:
        slug_risk = risk_limit_overrides.get(slug)
        if isinstance(slug_risk, dict) and slug_risk:
            next_risk_limits = deep_merge(next_risk_limits, slug_risk)
    if next_risk_limits != risk_limits:
        changed = True

    if not changed:
        return {
            "action": f"apply_trader_overrides:{trader_id}",
            "ok": True,
            "result": "already_current",
        }

    result = safe_call(
        f"update_trader:{trader_id}",
        lambda: api.put(
            f"/api/traders/{trader_id}",
            {
                "source_configs": source_configs,
                "risk_limits": next_risk_limits,
                "requested_by": operator,
                "reason": "caretaker live managed trader constraints",
            },
        ),
    )
    return {
        "action": f"apply_trader_overrides:{trader_id}",
        "ok": bool(result.get("ok")),
        "result": summarize_trader_update_result(result),
    }


def build_backtest_windows(policy: dict[str, Any]) -> list[dict[str, str]]:
    research = policy.get("research") if isinstance(policy.get("research"), dict) else {}
    fixed = research.get("windows_utc")
    if isinstance(fixed, list) and fixed:
        windows = []
        for item in fixed:
            if not isinstance(item, dict):
                continue
            start = str(item.get("start") or "").strip()
            end = str(item.get("end") or "").strip()
            if start and end:
                windows.append({"start": start, "end": end})
        if windows:
            return windows

    rolling_hours = max(1, to_int(research.get("rolling_window_hours"), 3))
    window_hours = max(1, to_int(research.get("window_hours"), 1))
    exclude_recent = max(0, to_int(research.get("exclude_recent_minutes"), 15))
    end = utc_now().replace(minute=0, second=0, microsecond=0) - timedelta(minutes=exclude_recent)
    if end.minute != 0:
        end = end.replace(minute=0, second=0, microsecond=0)
    start_floor = end - timedelta(hours=rolling_hours)
    windows = []
    cursor = start_floor
    while cursor < end:
        nxt = min(cursor + timedelta(hours=window_hours), end)
        windows.append({"start": iso_utc(cursor), "end": iso_utc(nxt)})
        cursor = nxt
    return windows


def strategy_rows_by_slug(api: HomerunApiClient) -> dict[str, dict[str, Any]]:
    payload = api.get("/api/strategy-manager", {"enabled": "true"})
    rows = payload.get("items") if isinstance(payload, dict) else []
    out: dict[str, dict[str, Any]] = {}
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, dict):
            continue
        slug = str(row.get("slug") or "").strip()
        if slug:
            out[slug] = row
    return out


def build_backtest_variants(
    strategies: dict[str, dict[str, Any]],
    traders: list[dict[str, Any]],
    policy: dict[str, Any],
) -> list[dict[str, Any]]:
    research = policy.get("research") if isinstance(policy.get("research"), dict) else {}
    requested_slugs = [
        str(item).strip()
        for item in (research.get("candidate_strategy_slugs") or [])
        if str(item).strip()
    ]
    config_overrides = research.get("config_overrides") if isinstance(research.get("config_overrides"), dict) else {}
    variants: list[dict[str, Any]] = []
    for slug in requested_slugs:
        row = strategies.get(slug)
        if not row:
            continue
        source_code = str(row.get("source_code") or "")
        if len(source_code) < 10:
            continue
        base_config = row.get("config") if isinstance(row.get("config"), dict) else {}
        variants.append(
            {
                "variant_key": f"{slug}:strategy_manager",
                "strategy_id": row.get("id"),
                "strategy_slug": slug,
                "strategy_version": row.get("version"),
                "source_code": source_code,
                "config": base_config,
                "config_override": {},
                "source": "strategy_manager",
            }
        )
        for index, override in enumerate(config_overrides.get(slug) or [], start=1):
            if not isinstance(override, dict):
                continue
            variants.append(
                {
                    "variant_key": f"{slug}:override:{index}",
                    "strategy_id": row.get("id"),
                    "strategy_slug": slug,
                    "strategy_version": row.get("version"),
                    "source_code": source_code,
                    "config": deep_merge(base_config, override),
                    "config_override": override,
                    "source": "policy_override",
                }
            )

    if bool(research.get("include_running_trader_params", True)):
        seen: set[str] = {str(variant["variant_key"]) for variant in variants}
        for trader in traders:
            if not isinstance(trader, dict):
                continue
            if not bool(trader.get("is_enabled")) or bool(trader.get("is_paused")):
                continue
            for cfg in trader.get("source_configs") or []:
                if not isinstance(cfg, dict):
                    continue
                slug = str(cfg.get("strategy_key") or "").strip()
                if slug not in requested_slugs:
                    continue
                row = strategies.get(slug)
                if not row:
                    continue
                params = cfg.get("strategy_params") if isinstance(cfg.get("strategy_params"), dict) else {}
                if not params:
                    continue
                variant_key = f"{slug}:trader:{trader.get('id')}"
                if variant_key in seen:
                    continue
                seen.add(variant_key)
                variants.append(
                    {
                        "variant_key": variant_key,
                        "strategy_id": row.get("id"),
                        "strategy_slug": slug,
                        "strategy_version": row.get("version"),
                        "source_code": str(row.get("source_code") or ""),
                        "config": params,
                        "config_override": params,
                        "source": "running_trader",
                        "trader_id": trader.get("id"),
                        "trader_name": trader.get("name"),
                    }
                )
    return sorted(
        variants,
        key=lambda item: 0 if item.get("source") == "running_trader" else 1,
    )


def summarize_backtest_status(payload: dict[str, Any], variant: dict[str, Any], window: dict[str, str]) -> dict[str, Any]:
    return {
        "variant_key": variant.get("variant_key"),
        "strategy_slug": variant.get("strategy_slug"),
        "source": variant.get("source"),
        "trader_id": variant.get("trader_id"),
        "trader_name": variant.get("trader_name"),
        "window": window,
        "run_id": payload.get("run_id"),
        "status": payload.get("status"),
        "progress": payload.get("progress"),
        "message": payload.get("message"),
        "trade_count": to_int(payload.get("trade_count")),
        "total_return_pct": to_float(payload.get("total_return_pct")),
        "snapshots_processed": to_int(payload.get("snapshots_processed")),
        "snapshots_total_estimate": payload.get("snapshots_total_estimate"),
        "error": payload.get("error"),
    }


def enqueue_and_poll_backtest(
    api: HomerunApiClient,
    variant: dict[str, Any],
    window: dict[str, str],
    policy: dict[str, Any],
) -> dict[str, Any]:
    research = policy.get("research") if isinstance(policy.get("research"), dict) else {}
    slug = str(variant.get("strategy_slug") or "caretaker")
    run_slug = f"caretaker_{slug}_{int(time.time())}"
    payload = {
        "source_code": variant["source_code"],
        "slug": run_slug,
        "config": variant.get("config") or {},
        "start": window["start"],
        "end": window["end"],
        "initial_capital_usd": to_float(research.get("initial_capital_usd"), 1000.0),
        "counterfactual_sample_size": to_int(research.get("counterfactual_sample_size"), 2),
        "ensemble_sample_size": to_int(research.get("ensemble_sample_size"), 2),
        "fills_sample_size": to_int(research.get("fills_sample_size"), 1000),
    }
    queued = api.post("/api/backtest/runs/enqueue", payload)
    run_id = str(queued.get("run_id") or "")
    if not run_id:
        raise RuntimeError(f"Backtest enqueue returned no run_id: {queued}")

    timeout_seconds = max(1, to_int(research.get("poll_timeout_seconds"), 1800))
    poll_interval = max(1.0, to_float(research.get("poll_interval_seconds"), 5.0))
    deadline = time.monotonic() + timeout_seconds
    last_status: dict[str, Any] = dict(queued)
    while time.monotonic() < deadline:
        status = api.get(f"/api/backtest/runs/{run_id}/status")
        last_status = status if isinstance(status, dict) else {"status": "unknown"}
        if str(last_status.get("status") or "").lower() in TERMINAL_BACKTEST_STATUSES:
            return summarize_backtest_status(last_status, variant, window)
        time.sleep(poll_interval)

    return summarize_backtest_status(
        {
            **last_status,
            "status": "timeout",
            "error": f"Backtest did not finish within {timeout_seconds} seconds",
        },
        variant,
        window,
    )


def aggregate_backtests(results: list[dict[str, Any]], policy: dict[str, Any]) -> dict[str, Any]:
    promotion = policy.get("promotion") if isinstance(policy.get("promotion"), dict) else {}
    by_variant: dict[str, dict[str, Any]] = {}
    for result in results:
        key = str(result.get("variant_key") or "")
        if not key:
            continue
        row = by_variant.setdefault(
            key,
            {
                "variant_key": key,
                "strategy_slug": result.get("strategy_slug"),
                "source": result.get("source"),
                "trader_id": result.get("trader_id"),
                "trader_name": result.get("trader_name"),
                "windows": 0,
                "completed_windows": 0,
                "failed_windows": 0,
                "trade_count": 0,
                "total_return_pct_sum": 0.0,
                "runs": [],
            },
        )
        row["windows"] += 1
        status = str(result.get("status") or "").lower()
        if status == "completed":
            row["completed_windows"] += 1
        else:
            row["failed_windows"] += 1
        row["trade_count"] += to_int(result.get("trade_count"))
        row["total_return_pct_sum"] += to_float(result.get("total_return_pct"))
        row["runs"].append(result)

    min_return = to_float(promotion.get("min_total_return_pct"), 0.0)
    min_trades = to_int(promotion.get("min_trade_count"), 5)
    require_all_completed = bool(promotion.get("require_all_windows_completed", True))
    ranked = sorted(
        by_variant.values(),
        key=lambda item: (
            to_float(item.get("total_return_pct_sum")),
            to_int(item.get("trade_count")),
            -to_int(item.get("failed_windows")),
        ),
        reverse=True,
    )
    for item in ranked:
        passed = (
            to_float(item.get("total_return_pct_sum")) >= min_return
            and to_int(item.get("trade_count")) >= min_trades
            and (not require_all_completed or to_int(item.get("failed_windows")) == 0)
        )
        item["passed_gate"] = passed
        item["gate"] = {
            "min_total_return_pct": min_return,
            "min_trade_count": min_trades,
            "require_all_windows_completed": require_all_completed,
        }
    return {"ranked": ranked, "best": ranked[0] if ranked else None}


def run_research_cycle(api: HomerunApiClient, policy: dict[str, Any]) -> dict[str, Any]:
    research = policy.get("research") if isinstance(policy.get("research"), dict) else {}
    if not bool(research.get("enabled", True)):
        return {"status": "skipped", "reason": "research disabled by policy"}

    strategies = strategy_rows_by_slug(api)
    traders_payload = api.get("/api/traders")
    traders = summarize_traders(traders_payload)
    variants = build_backtest_variants(strategies, traders, policy)
    windows = build_backtest_windows(policy)
    max_runs = max(1, to_int(research.get("max_runs_per_cycle"), 3))
    results: list[dict[str, Any]] = []
    scheduled = 0

    for variant in variants:
        for window in windows:
            if scheduled >= max_runs:
                break
            scheduled += 1
            try:
                results.append(enqueue_and_poll_backtest(api, variant, window, policy))
            except Exception as exc:
                results.append(
                    {
                        "variant_key": variant.get("variant_key"),
                        "strategy_slug": variant.get("strategy_slug"),
                        "source": variant.get("source"),
                        "window": window,
                        "status": "failed",
                        "trade_count": 0,
                        "total_return_pct": 0.0,
                        "error": str(exc),
                    }
                )
        if scheduled >= max_runs:
            break

    aggregate = aggregate_backtests(results, policy)
    promotion_action = apply_research_promotion(api, aggregate, variants, traders, policy)
    return {
        "status": "completed",
        "scheduled_runs": scheduled,
        "candidate_count": len(variants),
        "window_count": len(windows),
        "results": results,
        "aggregate": aggregate,
        "recommendation": build_research_recommendation(aggregate),
        "promotion": promotion_action,
    }


def build_research_recommendation(aggregate: dict[str, Any]) -> dict[str, Any]:
    best = aggregate.get("best") if isinstance(aggregate.get("best"), dict) else None
    if not best:
        return {"action": "none", "reason": "No completed research candidates."}
    if best.get("passed_gate"):
        return {
            "action": "eligible_for_manual_review",
            "reason": "Best candidate passed the configured backtest gate.",
            "variant_key": best.get("variant_key"),
            "total_return_pct_sum": best.get("total_return_pct_sum"),
            "trade_count": best.get("trade_count"),
        }
    return {
        "action": "do_not_promote",
        "reason": "No candidate passed the configured backtest gate.",
        "best_variant_key": best.get("variant_key"),
        "best_total_return_pct_sum": best.get("total_return_pct_sum"),
        "best_trade_count": best.get("trade_count"),
    }


def apply_research_promotion(
    api: HomerunApiClient,
    aggregate: dict[str, Any],
    variants: list[dict[str, Any]],
    traders: list[dict[str, Any]],
    policy: dict[str, Any],
) -> dict[str, Any]:
    promotion = policy.get("promotion") if isinstance(policy.get("promotion"), dict) else {}
    if not bool(promotion.get("allow_strategy_updates", False)) and not bool(
        promotion.get("allow_trader_config_updates", False)
    ):
        return {"status": "disabled", "reason": "promotion updates disabled by policy"}

    best = aggregate.get("best") if isinstance(aggregate.get("best"), dict) else None
    if not best:
        return {"status": "skipped", "reason": "no best candidate"}
    if not bool(best.get("passed_gate")):
        return {"status": "skipped", "reason": "best candidate did not pass promotion gate"}

    variants_by_key = {str(variant.get("variant_key")): variant for variant in variants}
    best_key = str(best.get("variant_key") or "")
    best_variant = variants_by_key.get(best_key)
    if not best_variant:
        return {"status": "skipped", "reason": f"best variant not found: {best_key}"}

    strategy_slug = str(best_variant.get("strategy_slug") or "")
    manager_key = f"{strategy_slug}:strategy_manager"
    manager_rows = [
        row for row in aggregate.get("ranked", [])
        if isinstance(row, dict) and row.get("variant_key") == manager_key
    ]
    if manager_rows and best_key != manager_key:
        improvement = to_float(best.get("total_return_pct_sum")) - to_float(
            manager_rows[0].get("total_return_pct_sum")
        )
        min_improvement = to_float(promotion.get("min_return_improvement_pct"), 0.02)
        if improvement < min_improvement:
            return {
                "status": "skipped",
                "reason": "best candidate did not clear manager-config improvement gate",
                "improvement_pct": improvement,
                "required_improvement_pct": min_improvement,
            }

    actions: list[dict[str, Any]] = []
    operator = str(policy.get("operator") or "codex_caretaker")
    best_config = best_variant.get("config") if isinstance(best_variant.get("config"), dict) else {}

    if bool(promotion.get("allow_strategy_updates", False)):
        strategy_id = str(best_variant.get("strategy_id") or "")
        if strategy_id:
            result = safe_call(
                f"promote_strategy_config:{strategy_id}",
                lambda: api.put(
                    f"/api/strategy-manager/{strategy_id}",
                    {"config": best_config, "enabled": True},
                ),
            )
            actions.append(
                {
                    "action": "update_strategy_manager_config",
                    "strategy_id": strategy_id,
                    "strategy_slug": strategy_slug,
                    "ok": bool(result.get("ok")),
                    "result": summarize_strategy_update_result(result),
                }
            )
        else:
            actions.append(
                {
                    "action": "update_strategy_manager_config",
                    "strategy_slug": strategy_slug,
                    "ok": False,
                    "result": "best candidate has no strategy_id",
                }
            )

    if bool(promotion.get("allow_trader_config_updates", False)):
        target_ids = {
            str(item).strip()
            for item in (promotion.get("target_trader_ids") or [])
            if str(item).strip()
        }
        if not target_ids:
            actions.append(
                {
                    "action": "update_trader_configs",
                    "ok": False,
                    "result": "allow_trader_config_updates requires promotion.target_trader_ids",
                }
            )
        for trader in traders:
            trader_id = str(trader.get("id") or "")
            if trader_id not in target_ids:
                continue
            source_configs = []
            touched = False
            for cfg in trader.get("source_configs") or []:
                if not isinstance(cfg, dict):
                    continue
                next_cfg = dict(cfg)
                if str(next_cfg.get("strategy_key") or "") == strategy_slug:
                    next_cfg["strategy_params"] = best_config
                    touched = True
                source_configs.append(next_cfg)
            if not touched:
                actions.append(
                    {
                        "action": "update_trader_config",
                        "trader_id": trader_id,
                        "ok": False,
                        "result": f"trader does not use strategy {strategy_slug}",
                    }
                )
                continue
            result = safe_call(
                f"promote_trader_config:{trader_id}",
                lambda trader_id=trader_id, source_configs=source_configs: api.put(
                    f"/api/traders/{trader_id}",
                    {
                        "source_configs": source_configs,
                        "requested_by": operator,
                        "reason": f"caretaker promotion {best_key}",
                    },
                ),
            )
            actions.append(
                {
                    "action": "update_trader_config",
                    "trader_id": trader_id,
                    "strategy_slug": strategy_slug,
                    "ok": bool(result.get("ok")),
                    "result": summarize_trader_update_result(result),
                }
            )

    return {
        "status": "applied" if any(action.get("ok") for action in actions) else "attempted",
        "variant_key": best_key,
        "actions": actions,
    }


def summarize_strategy_update_result(result: dict[str, Any]) -> Any:
    if not result.get("ok"):
        return result
    data = result.get("data")
    if not isinstance(data, dict):
        return data
    return {
        "id": data.get("id"),
        "slug": data.get("slug"),
        "version": data.get("version"),
        "status": data.get("status"),
        "enabled": data.get("enabled"),
        "error_message": data.get("error_message"),
    }


def summarize_trader_update_result(result: dict[str, Any]) -> Any:
    if not result.get("ok"):
        return result
    data = result.get("data")
    if not isinstance(data, dict):
        return data
    return {
        "id": data.get("id"),
        "name": data.get("name"),
        "mode": data.get("mode"),
        "is_enabled": data.get("is_enabled"),
        "is_paused": data.get("is_paused"),
        "block_new_orders": data.get("block_new_orders"),
        "updated_at": data.get("updated_at"),
    }


def write_report(report: dict[str, Any], report_dir: Path) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = utc_now().strftime("%Y%m%dT%H%M%SZ")
    path = report_dir / f"caretaker_{timestamp}.json"
    payload = json.dumps(report, indent=2, sort_keys=True)
    path.write_text(payload, encoding="utf-8")
    latest = report_dir / "latest.json"
    latest.write_text(payload, encoding="utf-8")
    return path


def build_report(
    command: str,
    policy: dict[str, Any],
    state: dict[str, Any],
    breaches: list[RiskBreach],
    actions: list[dict[str, Any]],
    research: dict[str, Any] | None,
    *,
    dry_run: bool,
) -> dict[str, Any]:
    orch = ((state.get("orchestrator") or {}).get("data") or {})
    snapshot = orch.get("snapshot") if isinstance(orch, dict) and isinstance(orch.get("snapshot"), dict) else {}
    control = orch.get("control") if isinstance(orch, dict) and isinstance(orch.get("control"), dict) else {}
    return {
        "timestamp": iso_utc(utc_now()),
        "command": command,
        "dry_run": dry_run,
        "operator": policy.get("operator"),
        "summary": {
            "mode": control.get("mode"),
            "kill_switch": control.get("kill_switch"),
            "daily_pnl": snapshot.get("daily_pnl"),
            "gross_exposure_usd": snapshot.get("gross_exposure_usd"),
            "open_orders": snapshot.get("open_orders"),
            "traders_running": snapshot.get("traders_running"),
            "current_activity": snapshot.get("current_activity"),
        },
        "breaches": [asdict(breach) for breach in breaches],
        "actions": actions,
        "state": state,
        "research": research,
    }


def _order_net_pnl(order: dict[str, Any]) -> float | None:
    """Canonical net realized PnL for a terminal binary order (stdlib mirror of
    backend/utils/pnl.py — keep in sync). Win nets shares*$1-cost, loss -cost;
    early closes trust the stored value clamped to the feasible range."""
    status = str(order.get("status") or "")
    entry = to_float(order.get("entry_price"))
    cost = to_float(order.get("filled_notional_usd")) or to_float(order.get("notional_usd"))
    shares = to_float(order.get("filled_shares")) or (cost / entry if entry > 0 else 0.0)
    if cost <= 0.0 or shares <= 0.0:
        return None
    max_win = shares - cost
    if status == "resolved_win":
        return max_win
    if status == "resolved_loss":
        return -cost
    if status in ("closed_win", "closed_loss"):
        return max(-cost, min(max_win, to_float(order.get("actual_profit"))))
    return None


def run_operator_cycle(api: HomerunApiClient, policy: dict[str, Any], *, dry_run: bool) -> dict[str, Any]:
    """Periodic operator: enforce a multi-day cumulative-bleed kill on the
    managed canary (the per-trader risk limits only catch single-day loss) and
    keep realized PnL honest. Deterministic; safe to run on a scheduler."""
    canary = policy.get("managed_canary") if isinstance(policy.get("managed_canary"), dict) else {}
    operator = str(policy.get("operator") or "caretaker_operator")
    actions: list[dict[str, Any]] = []
    canary_net: float | None = None
    trader_id = str(canary.get("trader_id") or "").strip()
    kill_net = to_float(canary.get("kill_net_usd"), -8.0)

    if trader_id:
        orders_call = safe_call("canary_orders", lambda: api.get(f"/api/traders/{trader_id}/orders", {"limit": 500}))
        if orders_call.get("ok"):
            raw = orders_call.get("data") or {}
            items = raw.get("orders") if isinstance(raw, dict) else raw
            nets = [_order_net_pnl(o) for o in (items or []) if isinstance(o, dict)]
            nets = [n for n in nets if n is not None]
            canary_net = round(sum(nets), 4)
            resolved = len(nets)
            if canary_net <= kill_net and resolved >= 1:
                if dry_run:
                    actions.append({"action": "block_canary", "ok": True, "result": {"dry_run": True, "net": canary_net}})
                else:
                    res = safe_call(
                        "block_canary",
                        lambda: api.post(
                            f"/api/traders/{trader_id}/block-new-orders",
                            {"enabled": True, "requested_by": operator,
                             "reason": f"cumulative bleed net={canary_net} <= {kill_net}"},
                        ),
                    )
                    actions.append({"action": "block_canary", "ok": bool(res.get("ok")),
                                    "result": res.get("data") if res.get("ok") else res})

    # PnL hygiene: keep realized PnL (and thus daily_pnl the guardian reads) honest.
    repair_corrected = 0
    if not dry_run:
        dry = safe_call("repair_dry", lambda: api.post("/api/maintenance/repair-implausible-pnl", {"dry_run": True}))
        if dry.get("ok") and to_int((dry.get("data") or {}).get("corrected")) > 0:
            real = safe_call("repair_apply", lambda: api.post("/api/maintenance/repair-implausible-pnl", {"dry_run": False}))
            repair_corrected = to_int((real.get("data") or {}).get("corrected")) if real.get("ok") else 0
            actions.append({"action": "repair_pnl", "ok": real.get("ok"), "corrected": repair_corrected})

    return {"canary_net": canary_net, "kill_net": kill_net, "actions": actions, "repair_corrected": repair_corrected}


def append_operator_log(report_dir: Path, summary: dict[str, Any], op: dict[str, Any]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    line = (
        f"{iso_utc(utc_now())} | mode={summary.get('mode')} | daily_pnl={summary.get('daily_pnl')} "
        f"| canary_net={op.get('canary_net')} | actions={[a.get('action') for a in op.get('actions', [])]}\n"
    )
    with (report_dir / "operator_log.txt").open("a", encoding="utf-8") as fh:
        fh.write(line)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Homerun caretaker guard and research harness")
    parser.add_argument(
        "command",
        choices=("status", "guard", "research", "cycle", "operator"),
        help="status reads state, guard enforces risk, research queues backtests, cycle does guard plus optional research, operator runs the canary kill-rule + PnL hygiene",
    )
    parser.add_argument("--api-base-url", default=DEFAULT_API_BASE_URL)
    parser.add_argument("--policy", type=Path, default=None)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-enforce", action="store_true")
    parser.add_argument("--deep", action="store_true")
    parser.add_argument("--run-backtests", action="store_true")
    parser.add_argument("--fail-on-breach", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    policy = load_policy(args.policy)
    api = HomerunApiClient(args.api_base_url)

    include_strategies = args.command == "research" or bool(args.run_backtests)
    # guard/cycle/operator must always read the live balance so the account-equity floor is enforced.
    deep = bool(args.deep) or args.command in {"guard", "cycle", "operator"}
    state = collect_operational_state(api, include_strategies=include_strategies, deep=deep)
    breaches = evaluate_risk(state, policy)
    actions: list[dict[str, Any]] = []
    should_enforce = args.command in {"guard", "cycle", "operator"} and not bool(args.no_enforce)
    if should_enforce:
        actions = enforce_risk_breaches(api, state, breaches, policy, dry_run=bool(args.dry_run))

    if args.command in {"guard", "cycle"} and not breaches:
        actions.extend(maintain_live_runtime(api, state, policy, dry_run=bool(args.dry_run)))

    operator_result = None
    if args.command == "operator" and not breaches:
        operator_result = run_operator_cycle(api, policy, dry_run=bool(args.dry_run))
        actions.extend(operator_result.get("actions", []))

    research_result = None
    if args.command == "research" or (args.command == "cycle" and bool(args.run_backtests)):
        if breaches:
            research_result = {
                "status": "skipped",
                "reason": "risk breaches present; research backtests deferred until guard is clean",
            }
        else:
            research_result = run_research_cycle(api, policy)

    report = build_report(
        args.command,
        policy,
        state,
        breaches,
        actions,
        research_result,
        dry_run=bool(args.dry_run),
    )
    report_path = write_report(report, args.report_dir)

    summary = report["summary"]
    if args.command == "operator":
        append_operator_log(args.report_dir, summary, operator_result or {})
    print(
        json.dumps(
            {
                "status": "breach" if breaches else "ok",
                "report": str(report_path),
                "mode": summary.get("mode"),
                "daily_pnl": summary.get("daily_pnl"),
                "gross_exposure_usd": summary.get("gross_exposure_usd"),
                "open_orders": summary.get("open_orders"),
                "breach_count": len(breaches),
                "action_count": len(actions),
                "research_status": (research_result or {}).get("status") if research_result else None,
                "canary_net": (operator_result or {}).get("canary_net") if operator_result else None,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if breaches and bool(args.fail_on_breach):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
