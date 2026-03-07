from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import ExecutionSessionLeg, TraderOrder
from services.event_bus import event_bus
from services.live_execution_adapter import execute_live_order
from services.live_execution_service import live_execution_service
from services.signal_bus import set_trade_signal_status
from services.strategy_sdk import StrategySDK
from services.worker_state import _commit_with_retry
from services.trader_orchestrator.strategies.registry import get_strategy
from services.trader_orchestrator.execution_policies import (
    allocate_leg_notionals,
    execution_waves,
    normalize_execution_constraints,
    normalize_execution_policy,
    reprice_limit_price,
    requires_pair_lock,
    supports_reprice,
)
from services.trader_orchestrator.order_manager import (
    cancel_live_provider_order,
    submit_execution_wave,
)
from services.trader_orchestrator_state import (
    _extract_copy_source_wallet_from_payload,
    _extract_live_fill_metrics,
    _serialize_order,
    _sync_order_runtime_payload,
    create_execution_session,
    create_execution_session_event,
    create_execution_session_order,
    create_trader_order,
    get_execution_session_detail,
    list_active_execution_sessions,
    sync_trader_position_inventory,
    update_execution_leg,
    update_execution_session_status,
)
from utils.converters import safe_float, safe_int
from utils.utcnow import utcnow
import services.trader_hot_state as hot_state


def _iso_utc(value: datetime) -> str:
    dt = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_iso_utc(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _strategy_instance_for_execution(strategy_key: str) -> Any | None:
    key = str(strategy_key or "").strip().lower()
    if not key:
        return None
    try:
        return get_strategy(key)
    except Exception:
        return None


def _strategy_supports_entry_take_profit_exit(strategy_instance: Any) -> bool:
    if strategy_instance is None:
        return False
    declared = getattr(strategy_instance, "supports_entry_take_profit_exit", None)
    if isinstance(declared, bool):
        return declared
    if declared is not None:
        return bool(declared)
    defaults = getattr(strategy_instance, "default_config", None)
    if not isinstance(defaults, dict):
        return False
    return "preplace_take_profit_exit" in defaults or "live_preplace_take_profit_exit" in defaults


def _resolve_preplace_take_profit_exit(
    *,
    strategy_instance: Any,
    params: dict[str, Any],
) -> bool:
    if not _strategy_supports_entry_take_profit_exit(strategy_instance):
        return False
    default_enabled = False
    defaults = getattr(strategy_instance, "default_config", None)
    if isinstance(defaults, dict):
        default_enabled = StrategySDK.should_preplace_take_profit_exit(defaults, default_enabled=False)
    return StrategySDK.should_preplace_take_profit_exit(params, default_enabled=default_enabled)


def _execution_profile_for_signal(
    *,
    signal: Any,
    strategy_key: str,
    legs_count: int,
) -> dict[str, Any]:
    source_key = str(getattr(signal, "source", "") or "").strip().lower()
    signal_strategy_type = str(getattr(signal, "strategy_type", "") or "").strip().lower()
    strategy_slug = str(strategy_key or "").strip().lower()

    payload = getattr(signal, "payload_json", None)
    payload = payload if isinstance(payload, dict) else {}
    strategy_context = payload.get("strategy_context")
    if not isinstance(strategy_context, dict):
        strategy_context = payload.get("strategy_context_json")
    context_source = (
        str(strategy_context.get("source_key") or "").strip().lower() if isinstance(strategy_context, dict) else ""
    )

    is_traders = (
        source_key == "traders"
        or context_source == "traders"
        or signal_strategy_type.startswith("traders")
        or strategy_slug.startswith("traders")
    )
    if is_traders:
        return {
            "policy": "REPRICE_LOOP" if legs_count <= 1 else "SEQUENTIAL_HEDGE",
            "price_policy": "taker_limit",
            "time_in_force": "IOC",
            "constraints": {
                "max_unhedged_notional_usd": 0.0,
                "hedge_timeout_seconds": 12,
                "session_timeout_seconds": 240,
                "max_reprice_attempts": 2,
                "pair_lock": legs_count > 1,
                "leg_fill_tolerance_ratio": 0.02,
            },
        }
    return {
        "policy": "PARALLEL_MAKER" if legs_count > 1 else "SINGLE_LEG",
        "price_policy": "maker_limit",
        "time_in_force": "GTC",
        "constraints": {
            "max_unhedged_notional_usd": 0.0,
            "hedge_timeout_seconds": 20,
            "session_timeout_seconds": 300,
            "max_reprice_attempts": 3,
            "pair_lock": legs_count > 1,
            "leg_fill_tolerance_ratio": 0.02,
        },
    }


def _is_live_position_cap_skip(*, mode: str, status: str, error_message: str | None) -> bool:
    if str(mode or "").strip().lower() != "live":
        return False
    if str(status or "").strip().lower() != "failed":
        return False
    error_text = str(error_message or "").strip().lower()
    if not error_text:
        return False
    return "maximum open positions" in error_text and "reached" in error_text


@dataclass
class SessionExecutionResult:
    session_id: str
    status: str
    effective_price: float | None
    error_message: str | None
    orders_written: int
    payload: dict[str, Any]
    created_orders: list[dict[str, Any]] = field(default_factory=list)


class ExecutionSessionEngine:
    def __init__(self, db: AsyncSession):
        self.db = db

    def _build_plan(
        self,
        signal: Any,
        *,
        strategy_key: str,
        size_usd: float,
        risk_limits: dict[str, Any],
    ) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
        payload = getattr(signal, "payload_json", None)
        payload = payload if isinstance(payload, dict) else {}
        execution_plan = payload.get("execution_plan")
        if not isinstance(execution_plan, dict):
            execution_plan = {}
        legs = [leg for leg in (execution_plan.get("legs") or []) if isinstance(leg, dict)]
        profile = _execution_profile_for_signal(
            signal=signal,
            strategy_key=strategy_key,
            legs_count=max(1, len(legs)),
        )

        if not legs:
            signal_payload = payload if isinstance(payload, dict) else {}
            fallback_market_id = str(getattr(signal, "market_id", "") or "")
            fallback_market_question = str(getattr(signal, "market_question", "") or "")
            legs = [
                {
                    "leg_id": "leg_1",
                    "market_id": fallback_market_id,
                    "market_question": fallback_market_question,
                    "token_id": str(
                        signal_payload.get("selected_token_id") or signal_payload.get("token_id") or ""
                    ).strip()
                    or None,
                    "side": "buy",
                    "outcome": None,
                    "limit_price": safe_float(getattr(signal, "entry_price", None), None),
                    "price_policy": str(profile["price_policy"]),
                    "time_in_force": str(profile["time_in_force"]),
                    "notional_weight": 1.0,
                    "min_fill_ratio": 0.0,
                    "metadata": {},
                }
            ]
            profile = _execution_profile_for_signal(
                signal=signal,
                strategy_key=strategy_key,
                legs_count=1,
            )

        policy = normalize_execution_policy(execution_plan.get("policy") or profile["policy"], legs_count=len(legs))
        raw_constraints = execution_plan.get("constraints")
        raw_constraints = raw_constraints if isinstance(raw_constraints, dict) else {}
        constraints = normalize_execution_constraints(raw_constraints)
        profile_constraints = dict(profile["constraints"])
        if safe_float(constraints.get("max_unhedged_notional_usd"), 0.0) <= 0:
            constraints["max_unhedged_notional_usd"] = max(
                0.0,
                safe_float(
                    risk_limits.get("max_unhedged_notional_usd"),
                    safe_float(profile_constraints.get("max_unhedged_notional_usd"), 0.0),
                ),
            )
        if "pair_lock" not in raw_constraints:
            constraints["pair_lock"] = bool(
                risk_limits.get("pair_lock", profile_constraints.get("pair_lock", constraints.get("pair_lock", True)))
            )
        if "hedge_timeout_seconds" not in raw_constraints:
            constraints["hedge_timeout_seconds"] = max(
                1,
                safe_int(
                    risk_limits.get("hedge_timeout_seconds"),
                    safe_int(profile_constraints.get("hedge_timeout_seconds"), constraints["hedge_timeout_seconds"]),
                ),
            )
        if "session_timeout_seconds" not in raw_constraints:
            constraints["session_timeout_seconds"] = max(
                1,
                safe_int(
                    risk_limits.get("session_timeout_seconds"),
                    safe_int(
                        profile_constraints.get("session_timeout_seconds"), constraints["session_timeout_seconds"]
                    ),
                ),
            )
        if "max_reprice_attempts" not in raw_constraints:
            constraints["max_reprice_attempts"] = max(
                0,
                safe_int(
                    risk_limits.get("max_reprice_attempts"),
                    safe_int(profile_constraints.get("max_reprice_attempts"), constraints["max_reprice_attempts"]),
                ),
            )
        if "leg_fill_tolerance_ratio" not in raw_constraints:
            constraints["leg_fill_tolerance_ratio"] = max(
                0.0,
                min(
                    1.0,
                    safe_float(
                        risk_limits.get("leg_fill_tolerance_ratio"),
                        safe_float(
                            profile_constraints.get("leg_fill_tolerance_ratio"), constraints["leg_fill_tolerance_ratio"]
                        ),
                    ),
                ),
            )
        notionals = allocate_leg_notionals(size_usd, legs)
        for index, leg in enumerate(legs):
            leg["requested_notional_usd"] = float(max(0.0, notionals[index]))
            limit_price = safe_float(leg.get("limit_price"), None)
            if limit_price is not None and limit_price > 0:
                leg["requested_shares"] = float(leg["requested_notional_usd"] / max(0.0001, limit_price))
            else:
                leg["requested_shares"] = None

        plan = {
            "plan_id": str(execution_plan.get("plan_id") or ""),
            "policy": policy,
            "time_in_force": str(execution_plan.get("time_in_force") or profile["time_in_force"]),
            "legs": legs,
            "constraints": constraints,
            "metadata": dict(execution_plan.get("metadata") or {}),
        }
        return plan, legs, constraints

    async def _fetch_leg_rows(self, session_id: str) -> dict[str, ExecutionSessionLeg]:
        rows = (
            (await self.db.execute(select(ExecutionSessionLeg).where(ExecutionSessionLeg.session_id == session_id)))
            .scalars()
            .all()
        )
        by_leg_id: dict[str, ExecutionSessionLeg] = {}
        for row in rows:
            by_leg_id[str(row.leg_id)] = row
        return by_leg_id

    def _effective_price_from_leg_results(self, leg_results: list[dict[str, Any]]) -> float | None:
        numerator = 0.0
        denominator = 0.0
        for result in leg_results:
            price = safe_float(result.get("effective_price"), None)
            notional = safe_float(result.get("notional_usd"), 0.0)
            if price is None or price <= 0 or notional <= 0:
                continue
            numerator += price * notional
            denominator += notional
        if denominator <= 0:
            return None
        return numerator / denominator

    async def execute_signal(
        self,
        *,
        trader_id: str,
        signal: Any,
        decision_id: str,
        strategy_key: str,
        strategy_version: int | None,
        strategy_params: dict[str, Any],
        risk_limits: dict[str, Any],
        mode: str,
        size_usd: float,
        reason: str | None,
    ) -> SessionExecutionResult:
        plan, legs, constraints = self._build_plan(
            signal,
            strategy_key=strategy_key,
            size_usd=size_usd,
            risk_limits=risk_limits,
        )
        session_timeout_seconds = safe_int(
            constraints.get("session_timeout_seconds"),
            300,
        )
        expires_at = utcnow() + timedelta(seconds=max(1, session_timeout_seconds))
        session_row = await create_execution_session(
            self.db,
            trader_id=trader_id,
            signal=signal,
            decision_id=decision_id,
            strategy_key=strategy_key,
            strategy_version=strategy_version,
            mode=mode,
            policy=plan["policy"],
            plan_id=plan["plan_id"] or None,
            legs=legs,
            requested_notional_usd=size_usd,
            max_unhedged_notional_usd=safe_float(
                constraints.get("max_unhedged_notional_usd"),
                0.0,
            ),
            expires_at=expires_at,
            payload={
                "execution_plan": plan,
                "strategy_key": strategy_key,
                "strategy_version": int(strategy_version) if strategy_version is not None else None,
                "reason": reason,
            },
            trace_id=str(getattr(signal, "trace_id", "") or "") or None,
            commit=False,
        )

        await create_execution_session_event(
            self.db,
            session_id=session_row.id,
            event_type="session_created",
            message=f"Execution session created with {len(legs)} leg(s)",
            payload={"policy": plan["policy"], "mode": mode},
            commit=False,
        )
        await update_execution_session_status(
            self.db,
            session_id=session_row.id,
            status="placing",
            commit=False,
        )

        leg_rows = await self._fetch_leg_rows(session_row.id)
        leg_execution_records: list[dict[str, Any]] = []
        order_write_inputs: list[dict[str, Any]] = []
        created_order_records: list[dict[str, Any]] = []
        orders_written = 0
        failed_legs = 0
        skipped_legs = 0
        open_legs = 0
        completed_legs = 0
        skip_reasons: list[str] = []

        waves = execution_waves(plan["policy"], legs)
        reprice_enabled = supports_reprice(plan["policy"])
        max_reprice_attempts = safe_int(constraints.get("max_reprice_attempts"), 3)
        strategy_instance = _strategy_instance_for_execution(strategy_key)
        preplace_take_profit_exit = _resolve_preplace_take_profit_exit(
            strategy_instance=strategy_instance,
            params=dict(strategy_params or {}),
        )

        for wave_index, wave in enumerate(waves):
            wave_with_notionals = [(leg, safe_float(leg.get("requested_notional_usd"), 0.0)) for leg in wave]
            wave_results = await submit_execution_wave(
                mode=mode,
                signal=signal,
                legs_with_notionals=wave_with_notionals,
            )

            for result in wave_results:
                leg_id = str(result.leg_id)
                leg_payload = (
                    next((candidate for candidate in wave if str(candidate.get("leg_id")) == leg_id), None) or {}
                )
                leg_row = leg_rows.get(leg_id)
                if leg_row is None:
                    continue

                mapped_leg_status = "failed"
                normalized_status = str(result.status or "").strip().lower()
                if _is_live_position_cap_skip(
                    mode=mode,
                    status=normalized_status,
                    error_message=result.error_message,
                ):
                    normalized_status = "skipped"
                if normalized_status == "executed":
                    mapped_leg_status = "completed"
                    completed_legs += 1
                elif normalized_status in {"open", "submitted"}:
                    mapped_leg_status = "open"
                    open_legs += 1
                elif normalized_status == "skipped":
                    mapped_leg_status = "skipped"
                    skipped_legs += 1
                    skip_reason = str(result.error_message or "").strip()
                    if skip_reason:
                        skip_reasons.append(skip_reason)
                else:
                    failed_legs += 1

                if mapped_leg_status == "failed" and reprice_enabled and max_reprice_attempts > 0:
                    reprice_price = reprice_limit_price(
                        safe_float(leg_payload.get("limit_price"), None),
                        str(leg_payload.get("side") or "buy"),
                        1,
                    )
                    if reprice_price is not None:
                        leg_payload["limit_price"] = reprice_price
                        retry_result = (
                            await submit_execution_wave(
                                mode=mode,
                                signal=signal,
                                legs_with_notionals=[
                                    (leg_payload, safe_float(leg_payload.get("requested_notional_usd"), 0.0))
                                ],
                            )
                        )[0]
                        normalized_retry = str(retry_result.status or "").strip().lower()
                        if normalized_retry in {"executed", "open", "submitted"}:
                            result = retry_result
                            normalized_status = normalized_retry
                            mapped_leg_status = "completed" if normalized_retry == "executed" else "open"
                            failed_legs -= 1
                            if mapped_leg_status == "completed":
                                completed_legs += 1
                            else:
                                open_legs += 1
                            await create_execution_session_event(
                                self.db,
                                session_id=session_row.id,
                                leg_id=leg_row.id,
                                event_type="reprice_success",
                                message="Leg succeeded after reprice attempt",
                                payload={"new_limit_price": reprice_price},
                                commit=False,
                            )

                filled_notional = (
                    safe_float(result.notional_usd, 0.0)
                    if normalized_status in {"executed", "open", "submitted"}
                    else 0.0
                )
                filled_shares = (
                    safe_float(result.shares, 0.0) if normalized_status in {"executed", "open", "submitted"} else 0.0
                )

                await update_execution_leg(
                    self.db,
                    leg_row_id=leg_row.id,
                    status=mapped_leg_status,
                    filled_notional_usd=filled_notional,
                    filled_shares=filled_shares,
                    avg_fill_price=safe_float(result.effective_price, None),
                    last_error=result.error_message if mapped_leg_status in {"failed", "skipped"} else None,
                    metadata_patch={
                        "wave_index": wave_index,
                        "provider_order_id": result.provider_order_id,
                        "provider_clob_order_id": result.provider_clob_order_id,
                    },
                    commit=False,
                )

                order_payload = dict(result.payload or {})
                order_payload["execution_session"] = {
                    "session_id": session_row.id,
                    "leg_id": leg_row.id,
                    "leg_ref": leg_id,
                    "policy": plan["policy"],
                }
                order_payload["strategy_type"] = str(getattr(signal, "strategy_type", "") or "").strip().lower()
                order_payload["strategy_context"] = (
                    getattr(signal, "strategy_context_json", None) or getattr(signal, "strategy_context", None) or {}
                )
                # Propagate live market context to order payload for freshness tracking.
                _signal_payload = getattr(signal, "payload_json", None)
                if isinstance(_signal_payload, dict):
                    _live_market = _signal_payload.get("live_market")
                    if isinstance(_live_market, dict):
                        order_payload["live_market"] = dict(_live_market)
                params = dict(strategy_params or {})
                exit_config: dict[str, Any] = {}
                for param_key, param_value in params.items():
                    key_text = str(param_key or "").strip()
                    if not key_text:
                        continue
                    exit_config[key_text] = param_value
                    if key_text.startswith("live_") and len(key_text) > 5:
                        canonical_key = key_text[5:]
                        if canonical_key and canonical_key not in exit_config:
                            exit_config[canonical_key] = param_value
                exit_key_aliases = {
                    "rapid_take_profit_pct": ("live_rapid_take_profit_pct", "rapid_take_profit_pct"),
                    "rapid_exit_window_minutes": ("live_rapid_exit_window_minutes", "rapid_exit_window_minutes"),
                    "rapid_exit_min_increase_pct": (
                        "live_rapid_exit_min_increase_pct",
                        "rapid_exit_min_increase_pct",
                    ),
                    "rapid_exit_breakeven_buffer_pct": (
                        "live_rapid_exit_breakeven_buffer_pct",
                        "rapid_exit_breakeven_buffer_pct",
                    ),
                    "take_profit_pct": ("live_take_profit_pct", "take_profit_pct"),
                    "stop_loss_pct": ("live_stop_loss_pct", "stop_loss_pct"),
                    "stop_loss_policy": ("live_stop_loss_policy", "stop_loss_policy", "stop_loss_mode"),
                    "stop_loss_activation_seconds": (
                        "live_stop_loss_activation_seconds",
                        "stop_loss_activation_seconds",
                        "live_stop_loss_near_close_seconds",
                        "stop_loss_near_close_seconds",
                    ),
                    "trailing_stop_pct": ("live_trailing_stop_pct", "trailing_stop_pct"),
                    "trailing_stop_activation_profit_pct": (
                        "live_trailing_stop_activation_profit_pct",
                        "trailing_stop_activation_profit_pct",
                    ),
                    "max_hold_minutes": ("live_max_hold_minutes", "max_hold_minutes"),
                    "min_hold_minutes": ("live_min_hold_minutes", "min_hold_minutes"),
                    "resolve_only": ("live_resolve_only", "resolve_only"),
                    "close_on_inactive_market": ("live_close_on_inactive_market", "close_on_inactive_market"),
                }
                for base_key in (
                    "rapid_take_profit_pct",
                    "rapid_exit_window_minutes",
                    "take_profit_pct",
                    "stop_loss_pct",
                    "stop_loss_policy",
                    "stop_loss_activation_seconds",
                    "trailing_stop_pct",
                    "trailing_stop_activation_profit_pct",
                    "max_hold_minutes",
                    "min_hold_minutes",
                ):
                    for suffix in ("5m", "15m", "1h", "4h"):
                        scoped_key = f"{base_key}_{suffix}"
                        exit_key_aliases[scoped_key] = (f"live_{scoped_key}", scoped_key)
                for target_key, aliases in exit_key_aliases.items():
                    selected_value = None
                    for alias_key in aliases:
                        if alias_key in params:
                            selected_value = params.get(alias_key)
                            break
                    if selected_value is not None:
                        exit_config[target_key] = selected_value

                order_payload["strategy_exit_config"] = exit_config
                if normalized_status != "skipped":
                    order_write_inputs.append(
                        {
                            "leg_id": leg_id,
                            "leg_row_id": leg_row.id,
                            "leg_payload": dict(leg_payload),
                            "normalized_status": normalized_status,
                            "result": result,
                            "order_payload": order_payload,
                            "exit_config": exit_config,
                        }
                    )

                leg_execution_records.append(
                    {
                        "leg_id": leg_id,
                        "status": normalized_status,
                        "effective_price": safe_float(result.effective_price, None),
                        "notional_usd": safe_float(result.notional_usd, 0.0),
                    }
                )

        session_detail = await get_execution_session_detail(self.db, session_row.id)
        session_view = session_detail["session"] if session_detail else {"unhedged_notional_usd": 0.0}
        pair_lock_enabled = requires_pair_lock(plan["policy"], constraints)
        max_unhedged = safe_float(constraints.get("max_unhedged_notional_usd"), 0.0)
        if pair_lock_enabled and max_unhedged > 0:
            current_unhedged = safe_float(session_view.get("unhedged_notional_usd"), 0.0)
            if current_unhedged > max_unhedged:
                violation_reason = (
                    f"Pair lock violation: unhedged notional {current_unhedged:.2f} exceeded {max_unhedged:.2f}."
                )
                await create_execution_session_event(
                    self.db,
                    session_id=session_row.id,
                    event_type="pair_lock_violation",
                    severity="warn",
                    message=violation_reason,
                    payload={
                        "current_unhedged_notional_usd": current_unhedged,
                        "max_unhedged_notional_usd": max_unhedged,
                    },
                    commit=False,
                )
                cancel_attempted_ids: set[str] = set()
                cancelled_provider_orders: list[str] = []
                cancel_failed_provider_orders: list[str] = []
                for item in order_write_inputs:
                    normalized_status = str(item.get("normalized_status") or "").strip().lower()
                    if normalized_status not in {"open", "submitted"}:
                        continue
                    result = item.get("result")
                    candidate_ids = [
                        str(getattr(result, "provider_clob_order_id", "") or "").strip(),
                        str(getattr(result, "provider_order_id", "") or "").strip(),
                    ]
                    cancelled = False
                    for candidate_id in candidate_ids:
                        if not candidate_id or candidate_id in cancel_attempted_ids:
                            continue
                        cancel_attempted_ids.add(candidate_id)
                        if await cancel_live_provider_order(candidate_id):
                            cancelled = True
                            cancelled_provider_orders.append(candidate_id)
                            break
                    if not cancelled and any(candidate_ids):
                        cancel_failed_provider_orders.extend([cid for cid in candidate_ids if cid])
                    leg_row_id = str(item.get("leg_row_id") or "").strip()
                    if leg_row_id:
                        await update_execution_leg(
                            self.db,
                            leg_row_id=leg_row_id,
                            status="cancelled",
                            last_error=violation_reason,
                            metadata_patch={"pair_lock_violation": True},
                            commit=False,
                        )
                    leg_id = str(item.get("leg_id") or "").strip()
                    for record in leg_execution_records:
                        if str(record.get("leg_id") or "") != leg_id:
                            continue
                        if str(record.get("status") or "").strip().lower() in {"open", "submitted"}:
                            record["status"] = "cancelled"
                        break

                effective_price = self._effective_price_from_leg_results(leg_execution_records)
                await update_execution_session_status(
                    self.db,
                    session_id=session_row.id,
                    status="failed",
                    error_message=violation_reason,
                    payload_patch={
                        "orders_written": 0,
                        "pair_lock_violation": {
                            "current_unhedged_notional_usd": current_unhedged,
                            "max_unhedged_notional_usd": max_unhedged,
                            "cancelled_provider_orders": cancelled_provider_orders,
                            "cancel_failed_provider_orders": cancel_failed_provider_orders,
                        },
                    },
                    commit=False,
                )
                await set_trade_signal_status(
                    self.db,
                    signal_id=str(getattr(signal, "id", "")),
                    status="failed",
                    effective_price=effective_price,
                    commit=False,
                )
                await create_execution_session_event(
                    self.db,
                    session_id=session_row.id,
                    event_type="session_aborted",
                    severity="error",
                    message="Execution session aborted after pair lock violation",
                    payload={
                        "current_unhedged_notional_usd": current_unhedged,
                        "max_unhedged_notional_usd": max_unhedged,
                        "cancelled_provider_orders": cancelled_provider_orders,
                        "cancel_failed_provider_orders": cancel_failed_provider_orders,
                    },
                    commit=False,
                )
                await _commit_with_retry(self.db)
                return SessionExecutionResult(
                    session_id=session_row.id,
                    status="failed",
                    effective_price=effective_price,
                    error_message=violation_reason,
                    orders_written=0,
                    payload={
                        "execution_plan": plan,
                        "legs": leg_execution_records,
                        "pair_lock_violation": {
                            "current_unhedged_notional_usd": current_unhedged,
                            "max_unhedged_notional_usd": max_unhedged,
                            "cancelled_provider_orders": cancelled_provider_orders,
                            "cancel_failed_provider_orders": cancel_failed_provider_orders,
                        },
                    },
                    created_orders=created_order_records,
                )

        for item in order_write_inputs:
            result = item["result"]
            leg_payload = dict(item.get("leg_payload") or {})
            order_payload = dict(item.get("order_payload") or {})
            normalized_status = str(item.get("normalized_status") or "").strip().lower()
            exit_config = dict(item.get("exit_config") or {})
            leg_row_id = str(item.get("leg_row_id") or "").strip()

            take_profit_pct = safe_float(exit_config.get("take_profit_pct"), None)
            entry_side = str(leg_payload.get("side") or "buy").strip().lower()
            if (
                mode == "live"
                and normalized_status == "executed"
                and entry_side == "buy"
                and preplace_take_profit_exit
                and take_profit_pct is not None
                and take_profit_pct > 0.0
            ):
                execution_payload = dict(result.payload or {})
                entry_fill_price = safe_float(result.effective_price, None)
                if entry_fill_price is None or entry_fill_price <= 0.0:
                    entry_fill_price = safe_float(execution_payload.get("average_fill_price"), None)
                exit_size = safe_float(execution_payload.get("filled_size"), None)
                if exit_size is None or exit_size <= 0.0:
                    exit_size = safe_float(result.shares, None)
                token_id = str(
                    execution_payload.get("token_id")
                    or (execution_payload.get("leg") or {}).get("token_id")
                    or leg_payload.get("token_id")
                    or ""
                ).strip()
                if (
                    token_id
                    and entry_fill_price is not None
                    and entry_fill_price > 0.0
                    and exit_size is not None
                    and exit_size > 0.0
                ):
                    target_price = min(0.999, max(0.001, entry_fill_price * (1.0 + (take_profit_pct / 100.0))))
                    try:
                        await live_execution_service.prepare_sell_balance_allowance(token_id)
                    except Exception:
                        pass
                    tp_submit = await execute_live_order(
                        token_id=token_id,
                        side="SELL",
                        size=float(exit_size),
                        fallback_price=target_price,
                        market_question=str(
                            leg_payload.get("market_question") or getattr(signal, "market_question", "") or ""
                        ),
                        opportunity_id=str(getattr(signal, "id", "") or ""),
                        time_in_force="GTC",
                        post_only=True,
                        resolve_live_price=False,
                    )
                    bracket_timestamp = _iso_utc(utcnow())
                    pending_exit: dict[str, Any] = {
                        "kind": "take_profit_limit",
                        "triggered_at": bracket_timestamp,
                        "last_attempt_at": bracket_timestamp,
                        "close_trigger": "take_profit_limit",
                        "close_price": float(target_price),
                        "price_source": "entry_bracket_limit",
                        "target_take_profit_pct": float(take_profit_pct),
                        "entry_fill_price": float(entry_fill_price),
                        "market_tradable": True,
                        "exit_size": float(exit_size),
                        "post_only": True,
                        "retry_count": 0,
                        "reason": "entry_bracket_take_profit",
                    }
                    if tp_submit.status in {"executed", "open", "submitted"}:
                        pending_exit["status"] = "submitted"
                        pending_exit["exit_order_id"] = str(tp_submit.order_id or "")
                        pending_exit["provider_clob_order_id"] = str(tp_submit.payload.get("clob_order_id") or "")
                        pending_exit["provider_status"] = str(tp_submit.payload.get("trading_status") or "")
                    else:
                        pending_exit["status"] = "failed"
                        pending_exit["retry_count"] = 1
                        pending_exit["last_error"] = str(tp_submit.error_message or "failed_to_place_take_profit_exit")
                    order_payload["pending_live_exit"] = pending_exit

            persisted_order_status = (
                "open" if mode in {"live", "shadow"} and normalized_status == "executed" else normalized_status
            )

            trader_order = await create_trader_order(
                self.db,
                trader_id=trader_id,
                signal=signal,
                decision_id=decision_id,
                strategy_key=strategy_key,
                strategy_version=strategy_version,
                mode=mode,
                status=persisted_order_status,
                notional_usd=safe_float(result.notional_usd, 0.0),
                effective_price=safe_float(result.effective_price, None),
                reason=reason,
                payload=order_payload,
                error_message=result.error_message,
                commit=False,
            )
            orders_written += 1
            created_order_records.append({
                "order_id": trader_order.id,
                "market_id": str(getattr(signal, "market_id", "") or ""),
                "direction": str(getattr(signal, "direction", "") or ""),
                "source": str(getattr(signal, "source", "") or ""),
                "notional_usd": safe_float(result.notional_usd, 0.0),
                "entry_price": safe_float(result.effective_price, 0.0),
                "token_id": str(leg_payload.get("token_id") or ""),
                "filled_shares": safe_float(result.shares, 0.0),
                "status": persisted_order_status,
                "payload": order_payload,
            })

            await create_execution_session_order(
                self.db,
                session_id=session_row.id,
                leg_id=leg_row_id,
                trader_order_id=trader_order.id,
                provider_order_id=result.provider_order_id,
                provider_clob_order_id=result.provider_clob_order_id,
                action="submit",
                side=str(leg_payload.get("side") or "buy"),
                price=safe_float(result.effective_price, None),
                size=safe_float(result.shares, None),
                notional_usd=safe_float(result.notional_usd, None),
                status=normalized_status,
                reason=reason,
                payload=result.payload,
                error_message=result.error_message,
                commit=False,
            )

        final_status = "completed"
        signal_status = "executed"
        error_message = None
        hedging_timeout_seconds: int | None = None
        hedging_started_at: datetime | None = None
        hedging_deadline_at: datetime | None = None
        if failed_legs == 0 and completed_legs == 0 and open_legs == 0 and skipped_legs > 0:
            final_status = "skipped"
            signal_status = "skipped"
            error_message = (
                skip_reasons[0]
                if skip_reasons
                else "Execution skipped before order submission."
            )
        elif failed_legs > 0 and completed_legs == 0 and open_legs == 0:
            final_status = "failed"
            signal_status = "failed"
            error_message = "All execution legs failed."
        elif failed_legs > 0:
            final_status = "hedging"
            signal_status = "submitted"
            error_message = "Partial fill: session in hedging state."
            hedging_timeout_seconds = max(1, safe_int(constraints.get("hedge_timeout_seconds"), 20))
            hedging_started_at = utcnow()
            hedging_deadline_at = hedging_started_at + timedelta(seconds=hedging_timeout_seconds)
        elif open_legs > 0:
            final_status = "working"
            signal_status = "submitted"

        effective_price = self._effective_price_from_leg_results(leg_execution_records)
        session_payload_patch: dict[str, Any] = {"orders_written": orders_written}
        if skipped_legs > 0:
            session_payload_patch["skipped_legs"] = skipped_legs
            if skip_reasons:
                session_payload_patch["skip_reasons"] = skip_reasons[:5]
        if final_status == "hedging" and hedging_started_at is not None and hedging_deadline_at is not None:
            session_payload_patch.update(
                {
                    "hedge_timeout_seconds": hedging_timeout_seconds,
                    "hedging_started_at": _iso_utc(hedging_started_at),
                    "hedging_deadline_at": _iso_utc(hedging_deadline_at),
                    "hedging_escalation": "auto_fail_on_timeout",
                }
            )
        await update_execution_session_status(
            self.db,
            session_id=session_row.id,
            status=final_status,
            error_message=error_message,
            payload_patch=session_payload_patch,
            commit=False,
        )
        await set_trade_signal_status(
            self.db,
            signal_id=str(getattr(signal, "id", "")),
            status=signal_status,
            effective_price=effective_price,
            commit=False,
        )
        event_type = "session_completed"
        event_severity = "info" if final_status == "completed" else "warn"
        event_message = f"Execution session ended with status={final_status}"
        if final_status in {"working", "hedging"}:
            event_type = "session_progress"
            event_severity = "info"
            event_message = f"Execution session remains {final_status}; awaiting leg completion"
        elif final_status == "skipped":
            event_type = "session_skipped"
            event_severity = "info"
            event_message = error_message or "Execution session skipped."
        await create_execution_session_event(
            self.db,
            session_id=session_row.id,
            event_type=event_type,
            severity=event_severity,
            message=event_message,
            payload={
                "failed_legs": failed_legs,
                "completed_legs": completed_legs,
                "open_legs": open_legs,
                "skipped_legs": skipped_legs,
                "hedge_timeout_seconds": hedging_timeout_seconds,
                "hedging_deadline_at": (_iso_utc(hedging_deadline_at) if hedging_deadline_at is not None else None),
            },
            commit=False,
        )
        await _commit_with_retry(self.db)

        return SessionExecutionResult(
            session_id=session_row.id,
            status=final_status,
            effective_price=effective_price,
            error_message=error_message,
            orders_written=orders_written,
            payload={
                "execution_plan": plan,
                "legs": leg_execution_records,
            },
            created_orders=created_order_records,
        )

    async def reconcile_active_sessions(
        self,
        *,
        mode: str,
        trader_id: str | None = None,
    ) -> dict[str, int]:
        sessions = await list_active_execution_sessions(
            self.db,
            trader_id=trader_id,
            mode=mode,
            limit=1000,
        )
        now = utcnow()
        expired = 0
        completed = 0
        cancelled = 0
        for row in sessions:
            status_key = str(row.status or "").strip().lower()
            if status_key == "paused":
                continue
            if row.expires_at is not None and now > row.expires_at:
                await self.cancel_session(
                    session_id=row.id,
                    reason="Session timed out before all legs completed.",
                    terminal_status="expired",
                )
                expired += 1
                continue

            if status_key == "hedging":
                session_payload = dict(row.payload_json or {}) if isinstance(row.payload_json, dict) else {}
                hedge_timeout_seconds = max(1, safe_int(session_payload.get("hedge_timeout_seconds"), 20))
                hedging_started_at = _parse_iso_utc(session_payload.get("hedging_started_at"))
                if hedging_started_at is None:
                    baseline_ts = row.updated_at or row.created_at or now
                    if baseline_ts.tzinfo is None:
                        hedging_started_at = baseline_ts.replace(tzinfo=timezone.utc)
                    else:
                        hedging_started_at = baseline_ts.astimezone(timezone.utc)
                hedging_deadline_at = _parse_iso_utc(session_payload.get("hedging_deadline_at"))
                if hedging_deadline_at is None:
                    hedging_deadline_at = hedging_started_at + timedelta(seconds=hedge_timeout_seconds)
                if now >= hedging_deadline_at:
                    timeout_reason = f"Hedge timeout exceeded ({hedge_timeout_seconds}s); escalating session to failed."
                    await create_execution_session_event(
                        self.db,
                        session_id=row.id,
                        event_type="hedge_timeout",
                        severity="error",
                        message=timeout_reason,
                        payload={
                            "hedge_timeout_seconds": hedge_timeout_seconds,
                            "hedging_started_at": _iso_utc(hedging_started_at),
                            "hedging_deadline_at": _iso_utc(hedging_deadline_at),
                        },
                        commit=False,
                    )
                    await self.cancel_session(
                        session_id=row.id,
                        reason=timeout_reason,
                        terminal_status="failed",
                    )
                    cancelled += 1
                    continue

            detail = await get_execution_session_detail(self.db, row.id)
            session_view = detail["session"] if detail else {}
            legs_total = safe_int(session_view.get("legs_total"), 0)
            legs_completed = safe_int(session_view.get("legs_completed"), 0)
            legs_failed = safe_int(session_view.get("legs_failed"), 0)
            legs_open = safe_int(session_view.get("legs_open"), 0)
            if legs_total > 0 and legs_completed >= legs_total:
                await update_execution_session_status(
                    self.db,
                    session_id=row.id,
                    status="completed",
                    commit=False,
                )
                completed += 1
            elif legs_failed > 0 and legs_open == 0 and legs_completed == 0:
                await update_execution_session_status(
                    self.db,
                    session_id=row.id,
                    status="failed",
                    error_message="All execution legs failed.",
                    commit=False,
                )
                cancelled += 1
        if expired > 0 or completed > 0 or cancelled > 0:
            await self.db.commit()
        return {
            "active_seen": len(sessions),
            "expired": expired,
            "completed": completed,
            "failed": cancelled,
        }

    async def cancel_session(
        self,
        *,
        session_id: str,
        reason: str,
        terminal_status: str = "cancelled",
    ) -> bool:
        detail = await get_execution_session_detail(self.db, session_id)
        if detail is None:
            return False
        session_view = detail["session"]
        terminal_key = str(terminal_status or "cancelled").strip().lower()
        if terminal_key not in {"cancelled", "expired", "failed"}:
            terminal_key = "cancelled"
        if str(session_view.get("status") or "").strip().lower() in {"cancelled", "completed", "failed", "expired"}:
            return True

        now = utcnow()

        # Batch-fetch live provider snapshots so we have up-to-date fill data
        # before deciding whether each order was filled or not.  Without this,
        # a session timeout can race with provider reconciliation and treat a
        # 99%-filled order as unfilled/cancelled.
        clob_ids_to_fetch: set[str] = set()
        for order in detail.get("orders", []):
            clob_id = str(order.get("provider_clob_order_id") or "").strip()
            if clob_id:
                clob_ids_to_fetch.add(clob_id)
        provider_snapshots: dict[str, dict[str, Any]] = {}
        if clob_ids_to_fetch:
            try:
                provider_snapshots = await live_execution_service.get_order_snapshots_by_clob_ids(
                    sorted(clob_ids_to_fetch)
                )
            except Exception:
                provider_snapshots = {}

        updated_trader_orders: list[TraderOrder] = []
        filled_leg_ids: set[str] = set()
        total_orders_processed = 0
        total_orders_filled = 0
        for order in detail.get("orders", []):
            status_key = str(order.get("status") or "").strip().lower()
            if status_key not in {"open", "submitted"}:
                continue
            trader_order_id = str(order.get("trader_order_id") or "").strip()
            trader_order = await self.db.get(TraderOrder, trader_order_id) if trader_order_id else None
            if trader_order is not None:
                local_status_key = str(trader_order.status or "").strip().lower()
                if local_status_key not in {"open", "submitted", "executed"}:
                    continue

            provider_order_id = str(order.get("provider_order_id") or "").strip()
            provider_clob_order_id = str(order.get("provider_clob_order_id") or "").strip()
            cancel_targets: list[str] = []
            if provider_clob_order_id:
                cancel_targets.append(provider_clob_order_id)
            if provider_order_id and provider_order_id not in cancel_targets:
                cancel_targets.append(provider_order_id)
            cancel_success = False
            for target in cancel_targets:
                if await cancel_live_provider_order(target):
                    cancel_success = True
                    break

            if trader_order is None:
                continue
            total_orders_processed += 1
            payload = dict(trader_order.payload_json or {})

            # Merge live provider snapshot into payload so _extract_live_fill_metrics
            # sees the latest fill data even if reconciliation hasn't written it yet.
            snapshot = provider_snapshots.get(provider_clob_order_id) if provider_clob_order_id else None
            if isinstance(snapshot, dict):
                existing_recon = payload.get("provider_reconciliation")
                if not isinstance(existing_recon, dict):
                    existing_recon = {}
                existing_recon = dict(existing_recon)
                existing_recon["snapshot"] = snapshot
                filled_size_snap = max(0.0, safe_float(snapshot.get("filled_size"), 0.0) or 0.0)
                avg_price_snap = safe_float(snapshot.get("average_fill_price"))
                filled_notional_snap = max(0.0, safe_float(snapshot.get("filled_notional_usd"), 0.0) or 0.0)
                if filled_notional_snap <= 0.0 and filled_size_snap > 0.0:
                    ref_price = avg_price_snap if avg_price_snap and avg_price_snap > 0 else safe_float(snapshot.get("limit_price"))
                    if ref_price and ref_price > 0:
                        filled_notional_snap = filled_size_snap * ref_price
                if filled_size_snap > 0.0:
                    existing_recon["filled_size"] = filled_size_snap
                if avg_price_snap is not None and avg_price_snap > 0:
                    existing_recon["average_fill_price"] = avg_price_snap
                if filled_notional_snap > 0.0:
                    existing_recon["filled_notional_usd"] = filled_notional_snap
                payload["provider_reconciliation"] = existing_recon

            filled_notional_usd, filled_shares, avg_fill_price = _extract_live_fill_metrics(payload)
            has_fill = filled_notional_usd > 0.0
            if has_fill:
                next_status = "executed"
                next_notional = max(0.0, filled_notional_usd)
                if next_notional <= 0.0:
                    next_notional = max(0.0, abs(safe_float(trader_order.notional_usd, 0.0) or 0.0))
                trader_order.status = next_status
                trader_order.notional_usd = float(next_notional)
                if avg_fill_price is not None and avg_fill_price > 0:
                    trader_order.effective_price = float(avg_fill_price)
                if trader_order.executed_at is None and next_notional > 0.0:
                    trader_order.executed_at = now
                total_orders_filled += 1
                leg_id = str(order.get("leg_id") or "").strip()
                if leg_id:
                    filled_leg_ids.add(leg_id)
            else:
                next_status = "cancelled"
                trader_order.status = next_status
                trader_order.notional_usd = 0.0

            payload["session_cancel"] = {
                "cancelled_at": _iso_utc(now),
                "session_id": session_id,
                "terminal_status": terminal_key,
                "reason": reason,
                "provider_cancel_targets": cancel_targets,
                "provider_cancel_success": bool(cancel_success),
            }
            trader_order.payload_json = _sync_order_runtime_payload(
                payload=payload,
                status=next_status,
                now=now,
                provider_snapshot_status=next_status if next_status == "cancelled" else None,
                mapped_status="cancelled" if next_status == "cancelled" else "executed",
            )
            trader_order.updated_at = now
            if reason:
                if trader_order.reason:
                    trader_order.reason = f"{trader_order.reason} | session:{terminal_key}:{reason}"
                else:
                    trader_order.reason = f"session:{terminal_key}:{reason}"
            if next_status == "cancelled":
                hot_state.record_order_cancelled(
                    trader_id=str(trader_order.trader_id or ""),
                    mode=str(trader_order.mode or ""),
                    order_id=str(trader_order.id or ""),
                    market_id=str(trader_order.market_id or ""),
                    source=str(trader_order.source or ""),
                    copy_source_wallet=_extract_copy_source_wallet_from_payload(payload),
                )
            updated_trader_orders.append(trader_order)

        any_filled = total_orders_filled > 0
        all_filled = total_orders_processed > 0 and total_orders_filled >= total_orders_processed

        for leg in detail.get("legs", []):
            leg_status = str(leg.get("status") or "").strip().lower()
            if leg_status in {"completed", "failed", "cancelled", "expired"}:
                continue
            leg_id = str(leg.get("id") or "").strip()
            if leg_id in filled_leg_ids:
                await update_execution_leg(
                    self.db,
                    leg_row_id=leg_id,
                    status="completed",
                    commit=False,
                )
            else:
                await update_execution_leg(
                    self.db,
                    leg_row_id=leg_id,
                    status="cancelled",
                    last_error=reason,
                    commit=False,
                )

        effective_session_status = "completed" if all_filled else terminal_key
        await update_execution_session_status(
            self.db,
            session_id=session_id,
            status=effective_session_status,
            error_message=None if all_filled else reason,
            commit=False,
        )
        # Any fill (even partial) means we hold a real position that must be
        # managed by the position lifecycle.  Mark the signal "executed" so the
        # downstream exit/P&L tracking picks it up.
        signal_id = str(session_view.get("signal_id") or "").strip()
        if signal_id:
            await set_trade_signal_status(
                self.db,
                signal_id=signal_id,
                status="executed" if any_filled else "failed",
                commit=False,
            )
        if all_filled:
            event_type = "session_completed"
            event_severity = "info"
        elif terminal_key == "expired":
            event_type = "session_expired"
            event_severity = "warn"
        elif terminal_key == "failed":
            event_type = "session_failed"
            event_severity = "error"
        else:
            event_type = "session_cancelled"
            event_severity = "warn"
        await create_execution_session_event(
            self.db,
            session_id=session_id,
            event_type=event_type,
            severity=event_severity,
            message=reason,
            commit=False,
        )
        await self.db.commit()

        # Trigger position inventory sync so filled orders are immediately
        # visible to the position lifecycle for exit management, rather than
        # waiting for the next worker loop iteration.
        if any_filled:
            trader_id = str(session_view.get("trader_id") or "").strip()
            if trader_id:
                try:
                    await sync_trader_position_inventory(
                        self.db,
                        trader_id=trader_id,
                        mode=str(session_view.get("mode") or ""),
                    )
                except Exception:
                    pass

        for row in updated_trader_orders:
            try:
                await event_bus.publish("trader_order", _serialize_order(row))
            except Exception:
                continue
        return True

    async def pause_session(self, *, session_id: str) -> bool:
        updated = await update_execution_session_status(
            self.db,
            session_id=session_id,
            status="paused",
            commit=False,
        )
        if updated is None:
            return False
        await create_execution_session_event(
            self.db,
            session_id=session_id,
            event_type="session_paused",
            message="Execution session paused",
            commit=False,
        )
        await self.db.commit()
        return True

    async def resume_session(self, *, session_id: str) -> bool:
        updated = await update_execution_session_status(
            self.db,
            session_id=session_id,
            status="working",
            commit=False,
        )
        if updated is None:
            return False
        await create_execution_session_event(
            self.db,
            session_id=session_id,
            event_type="session_resumed",
            message="Execution session resumed",
            commit=False,
        )
        await self.db.commit()
        return True
