from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import ExecutionSessionLeg
from services.signal_bus import set_trade_signal_status
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
    create_execution_session,
    create_execution_session_event,
    create_execution_session_order,
    create_trader_order,
    get_execution_session_detail,
    list_active_execution_sessions,
    update_execution_leg,
    update_execution_session_status,
)
from utils.converters import safe_float, safe_int
from utils.utcnow import utcnow


@dataclass
class SessionExecutionResult:
    session_id: str
    status: str
    effective_price: float | None
    error_message: str | None
    orders_written: int
    payload: dict[str, Any]


class ExecutionSessionEngine:
    def __init__(self, db: AsyncSession):
        self.db = db

    def _build_plan(
        self,
        signal: Any,
        *,
        size_usd: float,
        risk_limits: dict[str, Any],
    ) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
        payload = getattr(signal, "payload_json", None)
        payload = payload if isinstance(payload, dict) else {}
        execution_plan = payload.get("execution_plan")
        if not isinstance(execution_plan, dict):
            execution_plan = {}
        legs = [leg for leg in (execution_plan.get("legs") or []) if isinstance(leg, dict)]

        if not legs:
            signal_payload = payload if isinstance(payload, dict) else {}
            fallback_market_id = str(getattr(signal, "market_id", "") or "")
            fallback_market_question = str(getattr(signal, "market_question", "") or "")
            fallback_direction = str(getattr(signal, "direction", "") or "").strip().lower()
            fallback_outcome = (
                "yes" if fallback_direction == "buy_yes" else "no" if fallback_direction == "buy_no" else None
            )
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
                    "outcome": fallback_outcome,
                    "limit_price": safe_float(getattr(signal, "entry_price", None), None),
                    "price_policy": "maker_limit",
                    "time_in_force": "GTC",
                    "notional_weight": 1.0,
                    "min_fill_ratio": 0.0,
                    "metadata": {},
                }
            ]

        policy = normalize_execution_policy(execution_plan.get("policy"), legs_count=len(legs))
        raw_constraints = execution_plan.get("constraints")
        raw_constraints = raw_constraints if isinstance(raw_constraints, dict) else {}
        constraints = normalize_execution_constraints(raw_constraints)
        if safe_float(constraints.get("max_unhedged_notional_usd"), 0.0) <= 0:
            constraints["max_unhedged_notional_usd"] = max(
                0.0,
                safe_float(risk_limits.get("max_unhedged_notional_usd"), 0.0),
            )
        if "pair_lock" not in raw_constraints:
            constraints["pair_lock"] = bool(risk_limits.get("pair_lock", constraints.get("pair_lock", True)))
        if "hedge_timeout_seconds" not in raw_constraints:
            constraints["hedge_timeout_seconds"] = max(
                1,
                safe_int(risk_limits.get("hedge_timeout_seconds"), constraints["hedge_timeout_seconds"]),
            )
        if "session_timeout_seconds" not in raw_constraints:
            constraints["session_timeout_seconds"] = max(
                1,
                safe_int(risk_limits.get("session_timeout_seconds"), constraints["session_timeout_seconds"]),
            )
        if "max_reprice_attempts" not in raw_constraints:
            constraints["max_reprice_attempts"] = max(
                0,
                safe_int(risk_limits.get("max_reprice_attempts"), constraints["max_reprice_attempts"]),
            )
        if "leg_fill_tolerance_ratio" not in raw_constraints:
            constraints["leg_fill_tolerance_ratio"] = max(
                0.0,
                min(
                    1.0,
                    safe_float(
                        risk_limits.get("leg_fill_tolerance_ratio"),
                        constraints["leg_fill_tolerance_ratio"],
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
            "time_in_force": str(execution_plan.get("time_in_force") or "GTC"),
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
        strategy_params: dict[str, Any],
        risk_limits: dict[str, Any],
        mode: str,
        size_usd: float,
        reason: str | None,
    ) -> SessionExecutionResult:
        plan, legs, constraints = self._build_plan(
            signal,
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
        orders_written = 0
        failed_legs = 0
        open_legs = 0
        completed_legs = 0

        waves = execution_waves(plan["policy"], legs)
        reprice_enabled = supports_reprice(plan["policy"])
        max_reprice_attempts = safe_int(constraints.get("max_reprice_attempts"), 3)

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
                if normalized_status == "executed":
                    mapped_leg_status = "completed"
                    completed_legs += 1
                elif normalized_status in {"open", "submitted"}:
                    mapped_leg_status = "open"
                    open_legs += 1
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
                    last_error=result.error_message if mapped_leg_status == "failed" else None,
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
                exit_keys = {
                    "take_profit_pct",
                    "stop_loss_pct",
                    "trailing_stop_pct",
                    "max_hold_minutes",
                    "min_hold_minutes",
                    "resolve_only",
                    "close_on_inactive_market",
                }
                order_payload["strategy_exit_config"] = {
                    key: value for key, value in (strategy_params or {}).items() if key in exit_keys
                }

                trader_order = await create_trader_order(
                    self.db,
                    trader_id=trader_id,
                    signal=signal,
                    decision_id=decision_id,
                    mode=mode,
                    status=normalized_status,
                    notional_usd=safe_float(result.notional_usd, 0.0),
                    effective_price=safe_float(result.effective_price, None),
                    reason=reason,
                    payload=order_payload,
                    error_message=result.error_message,
                    commit=False,
                )
                orders_written += 1

                await create_execution_session_order(
                    self.db,
                    session_id=session_row.id,
                    leg_id=leg_row.id,
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
                await create_execution_session_event(
                    self.db,
                    session_id=session_row.id,
                    event_type="pair_lock_violation",
                    severity="warn",
                    message="Unhedged notional exceeded configured pair lock threshold",
                    payload={
                        "current_unhedged_notional_usd": current_unhedged,
                        "max_unhedged_notional_usd": max_unhedged,
                    },
                    commit=False,
                )
                failed_legs = max(failed_legs, 1)

        final_status = "completed"
        signal_status = "executed"
        error_message = None
        if failed_legs > 0 and completed_legs == 0 and open_legs == 0:
            final_status = "failed"
            signal_status = "failed"
            error_message = "All execution legs failed."
        elif failed_legs > 0:
            final_status = "hedging"
            signal_status = "submitted"
            error_message = "Partial fill: session in hedging state."
        elif open_legs > 0:
            final_status = "working"
            signal_status = "submitted"

        await update_execution_session_status(
            self.db,
            session_id=session_row.id,
            status=final_status,
            error_message=error_message,
            payload_patch={"orders_written": orders_written},
            commit=False,
        )
        await set_trade_signal_status(
            self.db,
            signal_id=str(getattr(signal, "id", "")),
            status=signal_status,
            effective_price=self._effective_price_from_leg_results(leg_execution_records),
            commit=False,
        )
        await create_execution_session_event(
            self.db,
            session_id=session_row.id,
            event_type="session_completed",
            severity="info" if final_status == "completed" else "warn",
            message=f"Execution session ended with status={final_status}",
            payload={
                "failed_legs": failed_legs,
                "completed_legs": completed_legs,
                "open_legs": open_legs,
            },
            commit=False,
        )
        await self.db.commit()

        effective_price = self._effective_price_from_leg_results(leg_execution_records)
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
                await update_execution_session_status(
                    self.db,
                    session_id=row.id,
                    status="expired",
                    error_message="Session timed out before all legs completed.",
                    commit=False,
                )
                await create_execution_session_event(
                    self.db,
                    session_id=row.id,
                    event_type="session_expired",
                    severity="warn",
                    message="Execution session expired by timeout policy",
                    commit=False,
                )
                expired += 1
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
    ) -> bool:
        detail = await get_execution_session_detail(self.db, session_id)
        if detail is None:
            return False
        session_view = detail["session"]
        if str(session_view.get("status") or "").strip().lower() in {"cancelled", "completed", "failed", "expired"}:
            return True

        for order in detail.get("orders", []):
            status_key = str(order.get("status") or "").strip().lower()
            if status_key not in {"open", "submitted"}:
                continue
            provider_order_id = str(order.get("provider_order_id") or "").strip()
            if provider_order_id:
                await cancel_live_provider_order(provider_order_id)

        for leg in detail.get("legs", []):
            leg_status = str(leg.get("status") or "").strip().lower()
            if leg_status in {"completed", "failed", "cancelled", "expired"}:
                continue
            await update_execution_leg(
                self.db,
                leg_row_id=str(leg["id"]),
                status="cancelled",
                last_error=reason,
                commit=False,
            )

        await update_execution_session_status(
            self.db,
            session_id=session_id,
            status="cancelled",
            error_message=reason,
            commit=False,
        )
        signal_id = str(session_view.get("signal_id") or "").strip()
        if signal_id:
            await set_trade_signal_status(
                self.db,
                signal_id=signal_id,
                status="failed",
                commit=False,
            )
        await create_execution_session_event(
            self.db,
            session_id=session_id,
            event_type="session_cancelled",
            severity="warn",
            message=reason,
            commit=False,
        )
        await self.db.commit()
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
