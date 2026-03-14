from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any

from services.live_execution_adapter import execute_live_order
from services.polymarket import polymarket_client
from services.live_execution_service import live_execution_service
from utils.converters import safe_float


_MIN_EXECUTION_PRICE = 0.001
_MIN_LIVE_SHARES = 5.0
_NUMERIC_TOKEN_ID_RE = re.compile(r"^\d{18,}$")
_HEX_TOKEN_ID_RE = re.compile(r"^(?:0x)?[0-9a-f]{40,}$")
_CONDITION_ID_RE = re.compile(r"^0x[0-9a-f]{64}$")


@dataclass
class LegSubmitResult:
    leg_id: str
    status: str
    effective_price: float | None
    error_message: str | None
    payload: dict[str, Any]
    provider_order_id: str | None = None
    provider_clob_order_id: str | None = None
    shares: float | None = None
    notional_usd: float | None = None


def _normalize_id(value: Any) -> str:
    return str(value or "").strip().lower()


def _looks_like_token_id(value: Any) -> bool:
    normalized = _normalize_id(value)
    if not normalized:
        return False
    if _CONDITION_ID_RE.fullmatch(normalized):
        return False
    return bool(_NUMERIC_TOKEN_ID_RE.fullmatch(normalized) or _HEX_TOKEN_ID_RE.fullmatch(normalized))


def _safe_signal_payload(signal: Any) -> dict[str, Any]:
    payload = getattr(signal, "payload_json", None)
    return payload if isinstance(payload, dict) else {}


def _safe_live_context(signal: Any, payload: dict[str, Any]) -> dict[str, Any]:
    context = getattr(signal, "live_context", None)
    if isinstance(context, dict):
        return context
    from_payload = payload.get("live_market")
    if isinstance(from_payload, dict):
        return from_payload
    return {}


def _resolve_token_id_for_leg(
    *,
    leg: dict[str, Any],
    payload: dict[str, Any],
    live_context: dict[str, Any],
) -> tuple[str | None, str | None, list[str]]:
    candidates: list[tuple[str, str]] = []

    def _append(source: str, value: Any) -> None:
        normalized = _normalize_id(value)
        if normalized:
            candidates.append((source, normalized))

    _append("leg.token_id", leg.get("token_id"))

    outcome = str(leg.get("outcome") or "").strip().lower()
    side = str(leg.get("side") or "buy").strip().lower()
    if outcome == "yes":
        _append("live_context.yes_token_id", live_context.get("yes_token_id"))
        _append("payload.yes_token_id", payload.get("yes_token_id"))
    elif outcome == "no":
        _append("live_context.no_token_id", live_context.get("no_token_id"))
        _append("payload.no_token_id", payload.get("no_token_id"))

    if side == "buy":
        _append("live_context.selected_token_id", live_context.get("selected_token_id"))
        _append("payload.selected_token_id", payload.get("selected_token_id"))

    token_ids = live_context.get("token_ids")
    if not isinstance(token_ids, list):
        token_ids = payload.get("token_ids")
    if isinstance(token_ids, list):
        for index, token in enumerate(token_ids):
            _append(f"token_ids[{index}]", token)

    _append("payload.token_id", payload.get("token_id"))

    for source, candidate in candidates:
        if _looks_like_token_id(candidate):
            return candidate, source, [entry[0] for entry in candidates]
    return None, None, [entry[0] for entry in candidates]


def _resolve_live_price_for_leg(leg: dict[str, Any], live_context: dict[str, Any]) -> float | None:
    if not isinstance(live_context, dict):
        return None

    selected_price = safe_float(live_context.get("live_selected_price"), None)
    yes_price = safe_float(live_context.get("live_yes_price"), None)
    no_price = safe_float(live_context.get("live_no_price"), None)

    def _valid(price: float | None) -> bool:
        return price is not None and price > 0

    leg_token = _normalize_id(leg.get("token_id"))
    selected_token = _normalize_id(live_context.get("selected_token_id"))
    yes_token = _normalize_id(live_context.get("yes_token_id"))
    no_token = _normalize_id(live_context.get("no_token_id"))
    outcome = str(leg.get("outcome") or "").strip().lower()
    selected_outcome = str(live_context.get("selected_outcome") or "").strip().lower()

    if leg_token:
        if selected_token and leg_token == selected_token and _valid(selected_price):
            return selected_price
        if yes_token and leg_token == yes_token and _valid(yes_price):
            return yes_price
        if no_token and leg_token == no_token and _valid(no_price):
            return no_price

    if outcome == "yes":
        if _valid(yes_price):
            return yes_price
        if selected_outcome == "yes" and _valid(selected_price):
            return selected_price
    if outcome == "no":
        if _valid(no_price):
            return no_price
        if selected_outcome == "no" and _valid(selected_price):
            return selected_price

    if not outcome and not leg_token and _valid(selected_price):
        return selected_price
    return None


def _resolve_leg_price(leg: dict[str, Any], signal: Any, live_context: dict[str, Any]) -> float | None:
    live_price = _resolve_live_price_for_leg(leg, live_context)
    if live_price is not None and live_price > 0:
        return live_price

    limit_price = safe_float(leg.get("limit_price"), None)
    if limit_price is not None and limit_price > 0:
        return limit_price

    signal_price = safe_float(getattr(signal, "entry_price", None), None)
    if signal_price is not None and signal_price > 0:
        return signal_price
    return None


def _resolve_condition_id_for_leg(
    *,
    leg: dict[str, Any],
    payload: dict[str, Any],
    live_context: dict[str, Any],
) -> str | None:
    metadata = leg.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    strategy_context = payload.get("strategy_context")
    strategy_context = strategy_context if isinstance(strategy_context, dict) else {}

    candidates = (
        leg.get("condition_id"),
        metadata.get("condition_id"),
        live_context.get("condition_id"),
        strategy_context.get("condition_id"),
        payload.get("condition_id"),
        leg.get("market_id"),
    )
    for raw in candidates:
        normalized = _normalize_id(raw)
        if normalized and _CONDITION_ID_RE.fullmatch(normalized):
            return normalized
    return None


async def _fetch_token_id_from_market(market_id: str, outcome: str) -> str | None:
    """Live fallback: query the Polymarket CLOB/Gamma API for token IDs by market condition_id."""
    condition_id = market_id.strip()
    if not condition_id:
        return None
    try:
        market_info = await polymarket_client.get_market_by_condition_id(condition_id)
    except Exception:
        return None
    if not isinstance(market_info, dict):
        return None

    for key in ("clobTokenIds", "clob_token_ids", "token_ids", "tokenIds"):
        raw = market_info.get(key)
        if isinstance(raw, list) and raw:
            token_ids = [str(t).strip() for t in raw if str(t).strip() and len(str(t).strip()) > 20]
            if not token_ids:
                continue
            outcome_lower = outcome.strip().lower()
            if outcome_lower == "yes" and len(token_ids) >= 1:
                return token_ids[0]
            if outcome_lower == "no" and len(token_ids) >= 2:
                return token_ids[1]
            return token_ids[0]
    return None


async def submit_execution_leg(
    *,
    mode: str,
    signal: Any,
    leg: dict[str, Any],
    notional_usd: float,
) -> LegSubmitResult:
    requested_mode_key = str(mode or "").strip().lower()
    mode_key = requested_mode_key
    if mode_key == "paper":
        mode_key = "shadow"
    legacy_paper_compat = requested_mode_key == "paper"
    if mode_key not in {"live", "shadow"}:
        return LegSubmitResult(
            leg_id=str(leg.get("leg_id") or "leg"),
            status="failed",
            effective_price=None,
            error_message=f"Unsupported execution mode '{mode_key or 'unknown'}'.",
            payload={"mode": mode_key or "unknown", "submission": "rejected", "reason": "unsupported_mode"},
            shares=None,
            notional_usd=float(max(0.0, notional_usd)),
        )
    leg_id = str(leg.get("leg_id") or "leg")
    notional = float(max(0.0, notional_usd))
    payload = _safe_signal_payload(signal)
    live_context = _safe_live_context(signal, payload)
    metadata = leg.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    side_key = str(leg.get("side") or "buy").strip().lower()
    ctf_action = str(metadata.get("ctf_action") or side_key).strip().lower()
    order_side = "SELL" if side_key == "sell" else "BUY"

    if ctf_action in {"split", "merge", "redeem"}:
        condition_id = _resolve_condition_id_for_leg(leg=leg, payload=payload, live_context=live_context)
        if not condition_id:
            return LegSubmitResult(
                leg_id=leg_id,
                status="failed",
                effective_price=None,
                error_message="Missing condition_id for CTF execution leg.",
                payload={
                    "mode": mode_key,
                    "leg": dict(leg),
                    "reason": "missing_condition_id",
                    "ctf_action": ctf_action,
                },
                shares=None,
                notional_usd=notional,
            )

        if mode_key != "live":
            return LegSubmitResult(
                leg_id=leg_id,
                status="executed",
                effective_price=None,
                error_message=None,
                payload={
                    "mode": mode_key,
                    "submission": "shadow_ctf_simulated",
                    "ctf_action": ctf_action,
                    "condition_id": condition_id,
                    "requested_notional_usd": notional,
                    "leg": dict(leg),
                },
                shares=None,
                notional_usd=notional,
            )

        from services.ctf_execution import ctf_execution_service

        if ctf_action == "split":
            amount_usd = max(
                0.0,
                safe_float(
                    metadata.get("amount_usd", leg.get("amount_usd")),
                    notional,
                )
                or 0.0,
            )
            ctf_result = await ctf_execution_service.split_position(
                condition_id=condition_id,
                amount_usd=amount_usd,
            )
            result_notional = amount_usd
        elif ctf_action == "merge":
            shares_per_side = max(
                0.0,
                safe_float(
                    metadata.get("shares_per_side", leg.get("shares_per_side")),
                    notional,
                )
                or 0.0,
            )
            ctf_result = await ctf_execution_service.merge_positions(
                condition_id=condition_id,
                shares_per_side=shares_per_side,
            )
            result_notional = shares_per_side
        else:
            raw_index_sets = metadata.get("index_sets", leg.get("index_sets"))
            index_sets: list[int] = []
            if isinstance(raw_index_sets, list):
                for value in raw_index_sets:
                    parsed = safe_float(value, None)
                    if parsed is None:
                        continue
                    as_int = int(parsed)
                    if as_int > 0:
                        index_sets.append(as_int)
            ctf_result = await ctf_execution_service.redeem_positions(
                condition_id=condition_id,
                index_sets=index_sets or None,
            )
            result_notional = notional

        normalized_status = "executed" if ctf_result.status == "executed" else "failed"
        return LegSubmitResult(
            leg_id=leg_id,
            status=normalized_status,
            effective_price=None,
            error_message=ctf_result.error_message,
            payload={
                "mode": mode_key,
                "submission": "ctf_execution",
                "ctf_action": ctf_action,
                "condition_id": condition_id,
                "tx_hash": ctf_result.tx_hash,
                "payload": dict(ctf_result.payload or {}),
                "leg": dict(leg),
            },
            provider_order_id=ctf_result.tx_hash,
            provider_clob_order_id=None,
            shares=None,
            notional_usd=result_notional,
        )

    price = _resolve_leg_price(leg, signal, live_context)

    if price is None or price <= 0:
        return LegSubmitResult(
            leg_id=leg_id,
            status="failed",
            effective_price=None,
            error_message="No valid price resolved for execution leg.",
            payload={"mode": mode_key, "leg": dict(leg), "reason": "missing_price"},
            shares=None,
            notional_usd=notional,
        )

    if price > 1.0:
        return LegSubmitResult(
            leg_id=leg_id,
            status="failed",
            effective_price=price,
            error_message="Execution price must be <= 1.0 for binary contracts.",
            payload={"mode": mode_key, "leg": dict(leg), "reason": "invalid_price_range"},
            shares=None,
            notional_usd=notional,
        )

    if price < _MIN_EXECUTION_PRICE:
        return LegSubmitResult(
            leg_id=leg_id,
            status="failed",
            effective_price=price,
            error_message=f"Execution price below minimum allowed ({_MIN_EXECUTION_PRICE:.4f}).",
            payload={"mode": mode_key, "leg": dict(leg), "reason": "invalid_price_too_small"},
            shares=None,
            notional_usd=notional,
        )

    requested_shares = notional / price
    if requested_shares <= 0:
        return LegSubmitResult(
            leg_id=leg_id,
            status="failed",
            effective_price=price,
            error_message="Computed leg size is zero.",
            payload={"mode": mode_key, "leg": dict(leg), "reason": "invalid_size"},
            shares=requested_shares,
            notional_usd=notional,
        )
    shares = requested_shares
    if shares < _MIN_LIVE_SHARES:
        shares = _MIN_LIVE_SHARES
    effective_notional = shares * price

    token_id, token_source, token_attempts = _resolve_token_id_for_leg(
        leg=leg,
        payload=payload,
        live_context=live_context,
    )
    if not token_id:
        market_id_for_lookup = str(leg.get("market_id") or "").strip()
        outcome_for_lookup = str(leg.get("outcome") or "").strip()
        if market_id_for_lookup and outcome_for_lookup:
            token_id = await _fetch_token_id_from_market(market_id_for_lookup, outcome_for_lookup)
            if token_id:
                token_source = "polymarket_api_fallback"
                token_attempts.append(token_source)
    if not token_id:
        if mode_key == "live":
            return LegSubmitResult(
                leg_id=leg_id,
                status="failed",
                effective_price=price,
                error_message="No executable token_id resolved for execution leg.",
                payload={
                    "mode": mode_key,
                    "submission": "rejected",
                    "reason": "missing_token_id",
                    "token_resolution_attempts": token_attempts,
                    "leg": dict(leg),
                },
                shares=shares,
                notional_usd=notional,
            )

    skip_buy_pre_submit_gate = False
    if mode_key == "live" and order_side == "BUY":
        buy_gate_ok, buy_gate_error = await live_execution_service.check_buy_pre_submit_gate(
            token_id=token_id,
            required_notional_usd=effective_notional,
        )
        if not buy_gate_ok:
            return LegSubmitResult(
                leg_id=leg_id,
                status="skipped",
                effective_price=price,
                error_message=buy_gate_error or "BUY pre-submit gate failed.",
                payload={
                    "mode": mode_key,
                    "submission": "skipped",
                    "reason": "buy_pre_submit_gate",
                    "token_id": token_id,
                    "token_id_source": token_source,
                    "token_resolution_attempts": token_attempts,
                    "leg": dict(leg),
                    "shares": shares,
                    "requested_shares": requested_shares,
                    "min_live_shares": _MIN_LIVE_SHARES,
                    "requested_notional_usd": notional,
                    "effective_notional_usd": effective_notional,
                },
                shares=shares,
                notional_usd=effective_notional,
            )
        skip_buy_pre_submit_gate = True

    time_in_force = str(leg.get("time_in_force") or "GTC").strip().upper()
    post_only = bool(leg.get("post_only", False))

    if mode_key == "shadow":
        quote_price = None
        quote_source = "signal_price_fallback"
        if token_id:
            try:
                quote_price = safe_float(await asyncio.wait_for(polymarket_client.get_midpoint(token_id), timeout=1.5), None)
            except Exception:
                quote_price = None
            quote_source = "polymarket_midpoint"
        if quote_price is None or quote_price <= 0.0 or quote_price > 1.0:
            quote_price = price
            if token_id:
                quote_source = "signal_price_fallback"
            else:
                quote_source = "signal_price_fallback_no_token"
        if quote_price is None or quote_price <= 0.0:
            return LegSubmitResult(
                leg_id=leg_id,
                status="failed",
                effective_price=price,
                error_message="No executable quote available for shadow execution leg.",
                payload={
                    "mode": "shadow",
                    "submission": "rejected",
                    "reason": "missing_quote",
                    "token_id": token_id,
                    "token_id_source": token_source,
                    "token_resolution_attempts": token_attempts,
                    "leg": dict(leg),
                },
                shares=shares,
                notional_usd=notional,
            )
        effective_shadow_notional = shares * quote_price
        payload_mode = "paper" if legacy_paper_compat else "shadow"
        submission_label = "simulated" if legacy_paper_compat else "shadow_quote_simulated"
        paper_fill_ratio = 1.0
        if notional > 0.0:
            paper_fill_ratio = min(1.0, max(0.0, effective_shadow_notional / notional))
        paper_simulation_payload = {
            "filled": True,
            "fill_ratio": paper_fill_ratio,
            "estimated_fee_usd": 0.0,
            "slippage_usd": max(0.0, notional - effective_shadow_notional),
        }
        return LegSubmitResult(
            leg_id=leg_id,
            status="executed",
            effective_price=quote_price,
            error_message=None,
            payload={
                "mode": payload_mode,
                "submission": submission_label,
                "token_id": token_id,
                "token_id_source": token_source,
                "token_resolution_attempts": token_attempts,
                "quote_source": quote_source,
                "quote_price": quote_price,
                "leg": dict(leg),
                "shares": shares,
                "filled_size": shares,
                "average_fill_price": quote_price,
                "filled_notional_usd": effective_shadow_notional,
                "requested_shares": requested_shares,
                "min_live_shares": _MIN_LIVE_SHARES,
                "requested_notional_usd": notional,
                "effective_notional_usd": effective_shadow_notional,
                "time_in_force": time_in_force,
                "post_only": post_only,
                "paper_simulation": paper_simulation_payload,
            },
            provider_order_id=None,
            provider_clob_order_id=None,
            shares=shares,
            notional_usd=effective_shadow_notional,
        )

    execution = await execute_live_order(
        token_id=token_id,
        side=order_side,
        size=shares,
        fallback_price=price,
        market_question=str(leg.get("market_question") or getattr(signal, "market_question", "") or ""),
        opportunity_id=str(getattr(signal, "id", "") or ""),
        time_in_force=time_in_force,
        post_only=post_only,
        enforce_fallback_bound=True,
        skip_buy_pre_submit_gate=skip_buy_pre_submit_gate,
    )

    execution_error_text = str(execution.error_message or "").lower()
    if (
        execution.status == "failed"
        and "orderbook" in execution_error_text
        and "does not exist" in execution_error_text
    ):
        market_id_for_lookup = str(leg.get("market_id") or "").strip()
        outcome_for_lookup = str(leg.get("outcome") or "").strip()
        if market_id_for_lookup and outcome_for_lookup:
            fallback_token_id = await _fetch_token_id_from_market(market_id_for_lookup, outcome_for_lookup)
            if fallback_token_id and fallback_token_id != token_id:
                retry_execution = await execute_live_order(
                    token_id=fallback_token_id,
                    side=order_side,
                    size=shares,
                    fallback_price=price,
                    market_question=str(leg.get("market_question") or getattr(signal, "market_question", "") or ""),
                    opportunity_id=str(getattr(signal, "id", "") or ""),
                    time_in_force=time_in_force,
                    post_only=post_only,
                    enforce_fallback_bound=True,
                    skip_buy_pre_submit_gate=skip_buy_pre_submit_gate,
                )
                if retry_execution.status != "failed":
                    execution = retry_execution
                    token_id = fallback_token_id
                    token_source = "polymarket_api_retry"
                else:
                    retry_error_text = str(retry_execution.error_message or "")
                    execution = retry_execution
                    execution.error_message = (
                        f"{str(execution.error_message or '')} | retry_token={fallback_token_id} failed: {retry_error_text}"
                    ).strip(" |")

    return LegSubmitResult(
        leg_id=leg_id,
        status=execution.status,
        effective_price=execution.effective_price,
        error_message=execution.error_message,
        payload={
            **execution.payload,
            "mode": "live",
            "leg": dict(leg),
            "token_id_source": token_source,
            "shares": shares,
            "requested_shares": requested_shares,
            "min_live_shares": _MIN_LIVE_SHARES,
            "requested_notional_usd": notional,
            "effective_notional_usd": effective_notional,
        },
        provider_order_id=execution.order_id,
        provider_clob_order_id=str(execution.payload.get("clob_order_id") or "").strip() or None,
        shares=shares,
        notional_usd=effective_notional,
    )


async def submit_execution_wave(
    *,
    mode: str,
    signal: Any,
    legs_with_notionals: list[tuple[dict[str, Any], float]],
) -> list[LegSubmitResult]:
    if not legs_with_notionals:
        return []
    tasks = [
        submit_execution_leg(
            mode=mode,
            signal=signal,
            leg=leg,
            notional_usd=notional,
        )
        for leg, notional in legs_with_notionals
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    normalized: list[LegSubmitResult] = []
    for index, result in enumerate(results):
        leg, notional = legs_with_notionals[index]
        leg_id = str(leg.get("leg_id") or f"leg_{index + 1}")
        if isinstance(result, Exception):
            normalized.append(
                LegSubmitResult(
                    leg_id=leg_id,
                    status="failed",
                    effective_price=safe_float(leg.get("limit_price"), None),
                    error_message=str(result),
                    payload={"mode": str(mode or "").lower(), "submission": "exception", "leg": dict(leg)},
                    shares=None,
                    notional_usd=notional,
                )
            )
            continue
        normalized.append(result)
    return normalized


async def cancel_live_provider_order(provider_order_id: str) -> bool:
    order_id = str(provider_order_id or "").strip()
    if not order_id:
        return False
    try:
        return bool(await live_execution_service.cancel_order(order_id))
    except Exception:
        return False
