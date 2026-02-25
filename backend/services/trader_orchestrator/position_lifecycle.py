from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from models.database import TradeSignal, TraderOrder
from services.polymarket import polymarket_client
from services.signal_bus import make_dedupe_key, refresh_trade_signal_snapshots, upsert_trade_signal
from services.simulation import simulation_service
from services.strategy_sdk import StrategySDK
from services.live_execution_service import live_execution_service
from utils.utcnow import utcnow
from utils.converters import safe_float

logger = logging.getLogger("position_lifecycle")

PAPER_ACTIVE_STATUSES = {"submitted", "executed", "open"}
LIVE_ACTIVE_STATUSES = {"submitted", "executed", "open"}
_FAILED_EXIT_MAX_RETRIES = 5
_FAILED_EXIT_MIN_RETRY_INTERVAL_SECONDS = 15
_WALLET_SIZE_EPSILON = 1e-9
_MARK_TOUCH_INTERVAL_SECONDS = 0.5
_MAX_LIVE_EXIT_FALLBACK_MARK_AGE_SECONDS = 20.0


def _iso_utc(value: datetime) -> str:
    dt = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _mark_touch_interval_seconds(params: dict[str, Any], *, mode: str) -> float:
    mode_key = str(mode or "").strip().lower()
    aliases: tuple[str, ...]
    if mode_key == "paper":
        aliases = ("paper_mark_touch_interval_seconds", "mark_touch_interval_seconds")
    else:
        aliases = ("live_mark_touch_interval_seconds", "mark_touch_interval_seconds")
    for key in aliases:
        parsed = safe_float(params.get(key))
        if parsed is None:
            continue
        return max(0.05, min(5.0, float(parsed)))
    return _MARK_TOUCH_INTERVAL_SECONDS


async def _publish_trader_order_updates(rows: list[TraderOrder]) -> None:
    if not rows:
        return
    from services.event_bus import event_bus
    from services.trader_orchestrator_state import _serialize_order

    seen: set[str] = set()
    for row in rows:
        row_id = str(getattr(row, "id", "") or "").strip()
        if not row_id or row_id in seen:
            continue
        seen.add(row_id)
        try:
            await event_bus.publish("trader_order", _serialize_order(row))
        except Exception:
            continue


def _failed_exit_retry_delay_seconds(last_error: Any) -> int:
    error_text = str(last_error or "").strip().lower()
    if "not enough balance / allowance" in error_text or "allowance" in error_text:
        return 8
    if "below minimum" in error_text or "exit_notional_below_min" in error_text:
        return 20
    if "missing token_id or fill_size" in error_text:
        return 30
    return _FAILED_EXIT_MIN_RETRY_INTERVAL_SECONDS


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _direction_outcome_index(direction: Any) -> Optional[int]:
    normalized = str(direction or "").strip().lower()
    if normalized == "buy_yes":
        return 0
    if normalized == "buy_no":
        return 1
    return None


def _opposite_direction(direction: Any) -> str:
    return StrategySDK.opposite_direction(direction, default="")


def _resolve_position_min_order_size_usd(
    *,
    trader_params: dict[str, Any],
    payload: dict[str, Any],
    mode: str,
) -> float:
    merged_params: dict[str, Any] = {}
    merged_params.update(dict(trader_params or {}))

    strategy_exit_config = payload.get("strategy_exit_config")
    if isinstance(strategy_exit_config, dict):
        merged_params.update(strategy_exit_config)

    strategy_context = payload.get("strategy_context")
    if isinstance(strategy_context, dict):
        strategy_context_params = strategy_context.get("params")
        if isinstance(strategy_context_params, dict):
            merged_params.update(strategy_context_params)

    payload_strategy_params = payload.get("strategy_params")
    if isinstance(payload_strategy_params, dict):
        merged_params.update(payload_strategy_params)

    return StrategySDK.resolve_min_order_size_usd(merged_params, mode=mode, fallback=1.0)


def _extract_market_token_ids(market_info: Optional[dict[str, Any]]) -> list[str]:
    if not isinstance(market_info, dict):
        return []
    token_ids = market_info.get("token_ids")
    if not isinstance(token_ids, list):
        token_ids = market_info.get("tokenIds")
    if not isinstance(token_ids, list):
        token_ids = market_info.get("clob_token_ids")
    if not isinstance(token_ids, list):
        token_ids = market_info.get("clobTokenIds")
    if not isinstance(token_ids, list):
        return []
    out: list[str] = []
    for token_id in token_ids:
        token_text = str(token_id or "").strip()
        if token_text:
            out.append(token_text)
    return out


def _extract_market_token_id(market_info: Optional[dict[str, Any]], outcome_idx: int) -> str:
    if outcome_idx not in (0, 1):
        return ""
    token_ids = _extract_market_token_ids(market_info)
    if not token_ids:
        return ""
    outcomes_raw = (market_info or {}).get("outcomes")
    if isinstance(outcomes_raw, list) and len(outcomes_raw) == len(token_ids):
        for idx, outcome_label in enumerate(outcomes_raw):
            label = str(outcome_label or "").strip().lower()
            if label == "yes" and outcome_idx == 0:
                return token_ids[idx]
            if label == "no" and outcome_idx == 1:
                return token_ids[idx]
    if len(token_ids) > outcome_idx:
        return token_ids[outcome_idx]
    return ""


def _normalize_reverse_entry(
    *,
    reverse_intent: Any,
    current_direction: str,
    current_price: Optional[float],
    market_seconds_left: Optional[float],
    market_info: Optional[dict[str, Any]],
    strategy_type: str,
    source: str,
    now: datetime,
) -> dict[str, Any] | None:
    normalized = StrategySDK.normalize_reverse_intent(
        reverse_intent,
        fallback_direction=_opposite_direction(current_direction),
        default_signal_type=f"{str(source or '').strip().lower() or 'strategy'}_reverse",
    )
    if not isinstance(normalized, dict):
        return None

    direction = str(normalized.get("direction") or "").strip().lower()
    if direction not in {"buy_yes", "buy_no"}:
        return None
    if direction == str(current_direction or "").strip().lower():
        return None

    min_seconds_left = max(0.0, safe_float(normalized.get("min_seconds_left"), 0.0) or 0.0)
    if market_seconds_left is not None and market_seconds_left < min_seconds_left:
        return None

    min_price_headroom = max(0.0, min(1.0, safe_float(normalized.get("min_price_headroom"), 0.0) or 0.0))
    if current_price is not None and current_price < min_price_headroom:
        return None

    outcome_idx = _direction_outcome_index(direction)
    if outcome_idx is None:
        return None

    entry_price = safe_float(normalized.get("entry_price"), None)
    if entry_price is None or entry_price <= 0.0 or entry_price >= 1.0:
        entry_price = _extract_market_side_price(market_info, outcome_idx)
    if (entry_price is None or entry_price <= 0.0 or entry_price >= 1.0) and current_price is not None:
        entry_price = 1.0 - current_price
    if entry_price is None or entry_price <= 0.0 or entry_price >= 1.0:
        return None

    token_id = str(normalized.get("token_id") or "").strip()
    if not token_id:
        token_id = _extract_market_token_id(market_info, outcome_idx)

    strategy_key = str(normalized.get("strategy_type") or strategy_type or "").strip().lower()
    signal_type = str(normalized.get("signal_type") or f"{source}_reverse").strip().lower()
    if not signal_type:
        signal_type = "strategy_reverse"

    expires_in_seconds = max(5.0, safe_float(normalized.get("expires_in_seconds"), 60.0) or 60.0)
    expires_at = now + timedelta(seconds=expires_in_seconds)
    out = dict(normalized)
    out["direction"] = direction
    out["entry_price"] = float(max(0.001, min(0.999, entry_price)))
    out["token_id"] = token_id
    out["strategy_type"] = strategy_key
    out["signal_type"] = signal_type
    out["source"] = str(source or "").strip().lower()
    out["expires_at"] = _iso_utc(expires_at)
    return out


def _arm_reverse_entry_from_exit(
    *,
    row: TraderOrder,
    payload: dict[str, Any],
    strategy_exit: Any,
    market_info: Optional[dict[str, Any]],
    market_seconds_left: Optional[float],
    current_price: Optional[float],
    now: datetime,
) -> dict[str, Any] | None:
    strategy_payload = getattr(strategy_exit, "payload", None)
    if not isinstance(strategy_payload, dict):
        return None
    reverse_intent = strategy_payload.get("reverse_intent")
    if not isinstance(reverse_intent, dict):
        return None

    existing = payload.get("pending_reverse_entry")
    existing = dict(existing) if isinstance(existing, dict) else {}
    existing_status = str(existing.get("status") or "").strip().lower()
    if existing_status in {"emitted", "submitted"} and str(existing.get("signal_id") or "").strip():
        return None

    prepared = _normalize_reverse_entry(
        reverse_intent=reverse_intent,
        current_direction=str(row.direction or ""),
        current_price=current_price,
        market_seconds_left=market_seconds_left,
        market_info=market_info,
        strategy_type=str(payload.get("strategy_type") or ""),
        source=str(row.source or ""),
        now=now,
    )
    if not isinstance(prepared, dict):
        return None

    attempt_index = int(safe_float(existing.get("attempt_index"), 0) or 0) + 1
    prepared["status"] = "armed"
    prepared["armed_at"] = _iso_utc(now)
    prepared["attempt_index"] = attempt_index
    prepared["source_order_id"] = str(row.id or "")
    prepared["source_signal_id"] = str(row.signal_id or "")
    prepared["trigger_reason"] = str(getattr(strategy_exit, "reason", "") or "").strip()
    payload["pending_reverse_entry"] = prepared
    return prepared


async def _emit_armed_reverse_signal(
    session: AsyncSession,
    *,
    row: TraderOrder,
    payload: dict[str, Any],
    signal_payload: dict[str, Any],
    market_info: Optional[dict[str, Any]],
    close_trigger: Optional[str],
    realized_pnl: Optional[float],
    now: datetime,
) -> tuple[str | None, str | None]:
    pending = payload.get("pending_reverse_entry")
    if not isinstance(pending, dict):
        return None, None
    status = str(pending.get("status") or "").strip().lower()
    if status not in {"armed", "failed", "ready"}:
        return None, None
    trigger_key = str(close_trigger or "").strip().lower()
    if "resolution" in trigger_key:
        pending["status"] = "skipped_resolution"
        pending["skipped_at"] = _iso_utc(now)
        payload["pending_reverse_entry"] = pending
        return None, None

    expires_at = _parse_iso_utc_naive(pending.get("expires_at"))
    now_naive = now.astimezone(timezone.utc).replace(tzinfo=None) if now.tzinfo is not None else now
    if expires_at is not None and now_naive > expires_at:
        pending["status"] = "expired"
        pending["expired_at"] = _iso_utc(now)
        payload["pending_reverse_entry"] = pending
        return None, None

    direction = str(pending.get("direction") or "").strip().lower()
    if direction not in {"buy_yes", "buy_no"}:
        pending["status"] = "failed"
        pending["last_error"] = "invalid_reverse_direction"
        pending["failed_at"] = _iso_utc(now)
        payload["pending_reverse_entry"] = pending
        return None, None
    outcome_idx = _direction_outcome_index(direction)
    if outcome_idx is None:
        pending["status"] = "failed"
        pending["last_error"] = "invalid_reverse_outcome_idx"
        pending["failed_at"] = _iso_utc(now)
        payload["pending_reverse_entry"] = pending
        return None, None

    entry_price = safe_float(pending.get("entry_price"), None)
    if entry_price is None or entry_price <= 0.0 or entry_price >= 1.0:
        entry_price = _extract_market_side_price(market_info, outcome_idx)
    if entry_price is None or entry_price <= 0.0 or entry_price >= 1.0:
        close_price = safe_float((payload.get("position_close") or {}).get("close_price"), None)
        if close_price is not None:
            entry_price = 1.0 - close_price
    if entry_price is None or entry_price <= 0.0 or entry_price >= 1.0:
        pending["status"] = "failed"
        pending["last_error"] = "missing_reverse_entry_price"
        pending["failed_at"] = _iso_utc(now)
        payload["pending_reverse_entry"] = pending
        return None, None

    token_id = str(pending.get("token_id") or "").strip()
    if not token_id:
        token_id = _extract_market_token_id(market_info, outcome_idx)
        if token_id:
            pending["token_id"] = token_id

    source = str(row.source or "").strip().lower() or "scanner"
    strategy_type = str(pending.get("strategy_type") or payload.get("strategy_type") or "").strip().lower() or None
    signal_type = str(pending.get("signal_type") or f"{source}_reverse").strip().lower()
    if not signal_type:
        signal_type = "strategy_reverse"
    confidence = safe_float(pending.get("confidence"), None)
    edge_percent = safe_float(pending.get("edge_percent"), None)
    liquidity = safe_float(
        pending.get("liquidity"),
        safe_float((signal_payload or {}).get("liquidity"), safe_float(getattr(row, "notional_usd", None), 0.0)),
    )

    payload_json = dict(signal_payload or {})
    payload_json["strategy_origin"] = str(payload_json.get("strategy_origin") or "crypto_worker")
    payload_json["reverse_entry"] = {
        "enabled": True,
        "source_order_id": str(row.id or ""),
        "source_signal_id": str(row.signal_id or ""),
        "close_trigger": trigger_key,
        "realized_pnl": realized_pnl,
        "direction": direction,
        "entry_price": float(entry_price),
        "size_multiplier": safe_float(pending.get("size_multiplier"), 1.0),
        "size_usd": safe_float(pending.get("size_usd")),
        "metadata": dict(pending.get("metadata") or {}) if isinstance(pending.get("metadata"), dict) else {},
    }
    payload_json["reverse_entry"]["emitted_at"] = _iso_utc(now)
    payload_json["selected_token_id"] = token_id or payload_json.get("selected_token_id")
    payload_json["token_id"] = token_id or payload_json.get("token_id")
    payload_json["signal_emitted_at"] = _iso_utc(now)
    payload_json["source_observed_at"] = _iso_utc(now)

    strategy_context = payload.get("strategy_context")
    strategy_context_json = dict(strategy_context) if isinstance(strategy_context, dict) else {}
    strategy_context_json["reverse_entry"] = dict(payload_json["reverse_entry"])
    strategy_context_json["source_order_id"] = str(row.id or "")
    strategy_context_json["source_signal_id"] = str(row.signal_id or "")
    strategy_context_json["selected_direction"] = direction
    if token_id:
        strategy_context_json["selected_token_id"] = token_id

    attempt_index = int(safe_float(pending.get("attempt_index"), 1) or 1)
    dedupe_key = make_dedupe_key(
        "reverse_entry",
        str(row.id or ""),
        attempt_index,
        direction,
        str(pending.get("armed_at") or ""),
    )
    source_item_id = str(pending.get("source_signal_id") or row.signal_id or row.id or "")
    market_question = (
        str(
            payload_json.get("market_question") or payload_json.get("question") or getattr(row, "market_id", "") or ""
        ).strip()
        or None
    )
    expires_at_dt = _parse_iso_utc_naive(pending.get("expires_at"))
    if expires_at_dt is None:
        expires_seconds = max(5.0, safe_float(pending.get("expires_in_seconds"), 60.0) or 60.0)
        expires_at_dt = now_naive + timedelta(seconds=expires_seconds)

    try:
        signal_row = await upsert_trade_signal(
            session,
            source=source,
            source_item_id=source_item_id,
            signal_type=signal_type,
            strategy_type=strategy_type,
            market_id=str(row.market_id or ""),
            market_question=market_question,
            direction=direction,
            entry_price=float(max(0.001, min(0.999, entry_price))),
            edge_percent=edge_percent,
            confidence=confidence,
            liquidity=liquidity,
            expires_at=expires_at_dt,
            payload_json=payload_json,
            strategy_context_json=strategy_context_json or None,
            dedupe_key=dedupe_key,
            commit=False,
        )
    except Exception as exc:
        pending["status"] = "failed"
        pending["last_error"] = str(exc)
        pending["failed_at"] = _iso_utc(now)
        payload["pending_reverse_entry"] = pending
        return None, None

    signal_id = str(getattr(signal_row, "id", "") or "").strip()
    pending["status"] = "emitted"
    pending["emitted_at"] = _iso_utc(now)
    pending["signal_id"] = signal_id
    pending["dedupe_key"] = dedupe_key
    payload["pending_reverse_entry"] = pending
    return signal_id or None, source


async def _publish_reverse_signal_batches(signal_ids_by_source: dict[str, list[str]], *, emitted_at: datetime) -> None:
    if not signal_ids_by_source:
        return
    from services.event_bus import event_bus

    emitted_at_iso = emitted_at.isoformat()
    for source_key, signal_ids in signal_ids_by_source.items():
        ids = [str(signal_id or "").strip() for signal_id in signal_ids if str(signal_id or "").strip()]
        if not ids:
            continue
        try:
            await event_bus.publish(
                "trade_signal_batch",
                {
                    "event_type": "reverse_entry",
                    "source": str(source_key or "").strip().lower(),
                    "signal_count": int(len(ids)),
                    "signal_ids": ids[:500],
                    "emitted_at": emitted_at_iso,
                    "trigger": "position_lifecycle_reverse",
                },
            )
        except Exception:
            continue


def _extract_signal_side_price(payload: dict[str, Any], outcome_idx: int) -> Optional[float]:
    side_keys = ("yes",) if outcome_idx == 0 else ("no",)
    for prefix in side_keys:
        for key in (
            f"{prefix}_price",
            f"{prefix}Price",
            f"best_{prefix}",
            f"best{prefix.title()}",
            f"{prefix}_mid",
            f"{prefix}Mid",
        ):
            parsed = safe_float(payload.get(key))
            if parsed is not None and parsed >= 0:
                return parsed

    prices = payload.get("outcome_prices")
    if not isinstance(prices, list):
        prices = payload.get("outcomePrices")
    if isinstance(prices, list) and len(prices) > outcome_idx:
        parsed = safe_float(prices[outcome_idx])
        if parsed is not None and parsed >= 0:
            return parsed
    return None


def _extract_market_side_price(market_info: Optional[dict[str, Any]], outcome_idx: int) -> Optional[float]:
    if not isinstance(market_info, dict):
        return None
    key = "yes_price" if outcome_idx == 0 else "no_price"
    parsed = safe_float(market_info.get(key))
    if parsed is not None and parsed >= 0:
        return parsed
    prices = market_info.get("outcome_prices")
    if isinstance(prices, list) and len(prices) > outcome_idx:
        parsed = safe_float(prices[outcome_idx])
        if parsed is not None and parsed >= 0:
            return parsed
    return None


def _extract_winning_outcome_index(market_info: Optional[dict[str, Any]]) -> Optional[int]:
    if not isinstance(market_info, dict):
        return None

    winner_raw = (
        market_info.get("winning_outcome")
        if market_info.get("winning_outcome") not in (None, "")
        else market_info.get("winner")
    )
    if winner_raw in (None, ""):
        return None

    outcomes_raw = market_info.get("outcomes")
    outcomes: list[str] = []
    if isinstance(outcomes_raw, list):
        outcomes = [str(item or "").strip().lower() for item in outcomes_raw if str(item or "").strip()]

    try:
        idx = int(winner_raw)
        if idx in (0, 1):
            return idx
    except Exception:
        pass

    winner_text = str(winner_raw).strip().lower()
    if winner_text == "yes":
        return 0
    if winner_text == "no":
        return 1
    if outcomes:
        for idx, label in enumerate(outcomes):
            if label == winner_text and idx in (0, 1):
                return idx
    return None


def _extract_winning_outcome_index_from_prices(
    market_info: Optional[dict[str, Any]],
    *,
    market_tradable: bool,
    settle_floor: float,
) -> Optional[int]:
    if not isinstance(market_info, dict):
        return None
    if market_tradable:
        return None

    settle_floor = min(1.0, max(0.5, settle_floor))
    settle_ceiling = max(0.0, 1.0 - settle_floor)

    yes_price = safe_float(market_info.get("yes_price"))
    no_price = safe_float(market_info.get("no_price"))
    if yes_price is None or no_price is None:
        prices = market_info.get("outcome_prices")
        if isinstance(prices, list) and len(prices) >= 2:
            outcomes = market_info.get("outcomes")
            if isinstance(outcomes, list) and len(outcomes) == len(prices):
                for idx, raw_label in enumerate(outcomes):
                    label = str(raw_label or "").strip().lower()
                    parsed = safe_float(prices[idx])
                    if parsed is None:
                        continue
                    if yes_price is None and label == "yes":
                        yes_price = parsed
                    if no_price is None and label == "no":
                        no_price = parsed
            if yes_price is None:
                yes_price = safe_float(prices[0])
            if no_price is None:
                no_price = safe_float(prices[1])

    yes_price = _state_price_floor(yes_price)
    no_price = _state_price_floor(no_price)
    if yes_price is None or no_price is None:
        return None
    if yes_price >= settle_floor and no_price <= settle_ceiling:
        return 0
    if no_price >= settle_floor and yes_price <= settle_ceiling:
        return 1
    return None


def _state_price_floor(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _status_for_close(*, pnl: float, close_trigger: Optional[str]) -> str:
    trigger = str(close_trigger or "").strip().lower()
    is_resolution = trigger in {"resolution", "resolution_inferred"}
    if is_resolution:
        return "resolved_win" if pnl >= 0 else "resolved_loss"
    return "closed_win" if pnl >= 0 else "closed_loss"


def _extract_position_state(payload: dict[str, Any]) -> dict[str, Any]:
    state = payload.get("position_state")
    return state if isinstance(state, dict) else {}


def _first_float_from_candidates(candidates: list[Any], keys: tuple[str, ...]) -> Optional[float]:
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        for key in keys:
            parsed = safe_float(candidate.get(key))
            if parsed is not None:
                return float(parsed)
    return None


def _extract_live_fill_metrics(payload: dict[str, Any]) -> tuple[float, float, Optional[float]]:
    provider_reconciliation = payload.get("provider_reconciliation")
    if not isinstance(provider_reconciliation, dict):
        provider_reconciliation = {}
    snapshot = provider_reconciliation.get("snapshot")
    if not isinstance(snapshot, dict):
        snapshot = {}
    candidates: list[Any] = [provider_reconciliation, snapshot, payload]
    filled_size = max(
        0.0,
        _first_float_from_candidates(
            candidates,
            (
                "filled_size",
                "size_matched",
                "sizeMatched",
                "matched_size",
                "filled_shares",
                "executed_size",
            ),
        )
        or 0.0,
    )
    average_fill_price = _first_float_from_candidates(
        candidates,
        (
            "average_fill_price",
            "avg_fill_price",
            "avg_price",
            "avgFillPrice",
            "matched_price",
            "price",
            "limit_price",
        ),
    )
    filled_notional = max(
        0.0,
        _first_float_from_candidates(
            candidates,
            (
                "filled_notional_usd",
                "filled_notional",
                "matched_notional",
                "matched_amount",
                "executed_notional",
            ),
        )
        or 0.0,
    )
    if filled_notional <= 0.0 and filled_size > 0.0 and average_fill_price is not None and average_fill_price > 0:
        filled_notional = filled_size * average_fill_price
    if filled_size <= 0.0 and filled_notional > 0.0 and average_fill_price is not None and average_fill_price > 0:
        filled_size = filled_notional / average_fill_price
    return filled_notional, filled_size, average_fill_price


def _extract_live_token_id(payload: dict[str, Any]) -> str:
    token_id = str(payload.get("token_id") or payload.get("selected_token_id") or "").strip()
    if token_id:
        return token_id
    provider_reconciliation = payload.get("provider_reconciliation")
    if isinstance(provider_reconciliation, dict):
        snapshot = provider_reconciliation.get("snapshot")
        if isinstance(snapshot, dict):
            token_id = str(snapshot.get("asset_id") or snapshot.get("asset") or snapshot.get("token_id") or "").strip()
            if token_id:
                return token_id
    return ""


def _extract_wallet_settlement_price(wallet_position: Optional[dict[str, Any]]) -> Optional[float]:
    if not isinstance(wallet_position, dict):
        return None
    if not _safe_bool(wallet_position.get("redeemable"), False):
        return None
    mark = safe_float(wallet_position.get("curPrice"))
    if mark is None:
        mark = safe_float(wallet_position.get("currentPrice"))
    if mark is None:
        return None
    if mark <= 0.001:
        return 0.0
    if mark >= 0.999:
        return 1.0
    return None


def _extract_wallet_position_size(wallet_position: Optional[dict[str, Any]]) -> float:
    if not isinstance(wallet_position, dict):
        return 0.0
    size = safe_float(wallet_position.get("size"))
    if size is None:
        size = safe_float(wallet_position.get("positionSize"))
    if size is None:
        size = safe_float(wallet_position.get("shares"))
    return max(0.0, float(size or 0.0))


def _extract_wallet_mark_price(wallet_position: Optional[dict[str, Any]]) -> Optional[float]:
    if not isinstance(wallet_position, dict):
        return None
    mark = safe_float(wallet_position.get("curPrice"))
    if mark is None:
        mark = safe_float(wallet_position.get("currentPrice"))
    if mark is None:
        mark = safe_float(wallet_position.get("markPrice"))
    if mark is None:
        mark = safe_float(wallet_position.get("price"))
    if mark is None or mark < 0:
        return None
    return float(mark)


def _pending_exit_fill_threshold(pending_exit: dict[str, Any]) -> float:
    parsed = safe_float(
        pending_exit.get("fill_ratio_threshold")
        if pending_exit.get("fill_ratio_threshold") is not None
        else pending_exit.get("fill_threshold_ratio")
    )
    if parsed is not None:
        return max(0.5, min(1.0, float(parsed)))

    required_exit_size = max(0.0, safe_float(pending_exit.get("exit_size"), 0.0) or 0.0)
    retry_count = int(safe_float(pending_exit.get("retry_count"), 0) or 0)
    close_trigger = str(pending_exit.get("close_trigger") or "").strip().lower()

    if required_exit_size <= 10.0:
        threshold = 0.88
    elif required_exit_size <= 25.0:
        threshold = 0.90
    elif required_exit_size <= 75.0:
        threshold = 0.93
    elif required_exit_size <= 200.0:
        threshold = 0.95
    else:
        threshold = 0.97

    if "stop_loss" in close_trigger or "rapid" in close_trigger:
        threshold = min(threshold, 0.92)
    if retry_count >= 2:
        threshold = max(0.85, threshold - 0.03)
    return max(0.5, min(1.0, float(threshold)))


def _remaining_exit_size(
    *,
    required_exit_size: float,
    pending_exit: dict[str, Any],
    wallet_position_size: float,
) -> float:
    required = max(0.0, float(required_exit_size))
    already_filled = max(0.0, safe_float(pending_exit.get("filled_size"), 0.0) or 0.0)
    remaining = max(0.0, required - already_filled)
    if required > 0.0 and remaining <= 0.0 and already_filled <= 0.0:
        remaining = required
    wallet_cap = wallet_position_size if wallet_position_size > _WALLET_SIZE_EPSILON else 0.0
    if wallet_cap > 0.0:
        if remaining <= 0.0:
            remaining = wallet_cap
        else:
            remaining = min(remaining, wallet_cap)
    return max(0.0, remaining)


def _pending_exit_provider_clob_id(pending_exit: dict[str, Any]) -> str:
    direct = str(pending_exit.get("provider_clob_order_id") or pending_exit.get("exit_order_clob_id") or "").strip()
    if direct:
        return direct
    fallback = str(pending_exit.get("exit_order_id") or "").strip()
    if not fallback:
        return ""
    if fallback.startswith("0x") or fallback.isdigit():
        return fallback
    try:
        cached = live_execution_service.get_order(fallback)
    except Exception:
        cached = None
    if cached is None:
        return ""
    return str(getattr(cached, "clob_order_id", "") or "").strip()


def _extract_wallet_trade_token_id(trade: dict[str, Any]) -> str:
    return str(
        trade.get("asset_id") or trade.get("asset") or trade.get("token_id") or trade.get("tokenId") or ""
    ).strip()


def _extract_wallet_trade_side(trade: dict[str, Any]) -> str:
    return str(trade.get("side") or trade.get("trade_side") or trade.get("type") or "").strip().lower()


def _extract_wallet_trade_size(trade: dict[str, Any]) -> float:
    size = safe_float(trade.get("size"))
    if size is None:
        size = safe_float(trade.get("amount"))
    if size is None:
        size = safe_float(trade.get("shares"))
    if size is None:
        size = safe_float(trade.get("matched_size"))
    return max(0.0, float(size or 0.0))


def _extract_wallet_trade_price(trade: dict[str, Any]) -> Optional[float]:
    price = safe_float(trade.get("price"))
    if price is None:
        price = safe_float(trade.get("avg_price"))
    if price is None:
        price = safe_float(trade.get("limit_price"))
    if price is None or price < 0:
        return None
    return float(price)


def _parse_iso_utc_naive(value: Any) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1]
    try:
        parsed = datetime.fromisoformat(text)
    except Exception:
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def _parse_wallet_trade_time(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            ts = float(value)
            if ts > 1e12:
                ts = ts / 1000.0
            return datetime.fromtimestamp(ts, tz=timezone.utc).replace(tzinfo=None)
        except Exception:
            return None
    text = str(value).strip()
    if not text:
        return None
    return _parse_iso_utc_naive(text)


def _parse_market_end_time_naive(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value
        return value.astimezone(timezone.utc).replace(tzinfo=None)
    if isinstance(value, (int, float)):
        try:
            ts = float(value)
            if ts > 1e12:
                ts = ts / 1000.0
            return datetime.fromtimestamp(ts, tz=timezone.utc).replace(tzinfo=None)
        except Exception:
            return None
    return _parse_iso_utc_naive(value)


def _extract_market_end_time_naive(market_info: Any) -> Optional[datetime]:
    if not isinstance(market_info, dict):
        return None
    for key in (
        "end_time",
        "endTime",
        "end_date",
        "endDate",
        "market_end_time",
        "expiration_time",
    ):
        parsed = _parse_market_end_time_naive(market_info.get(key))
        if parsed is not None:
            return parsed
    return None


def _market_seconds_left(market_info: Any, now: datetime) -> Optional[float]:
    if not isinstance(market_info, dict):
        return None
    for key in ("seconds_left", "secondsLeft", "seconds_until_close", "secondsUntilClose"):
        parsed = safe_float(market_info.get(key), None)
        if parsed is not None and parsed >= 0.0:
            return float(parsed)
    end_time = _extract_market_end_time_naive(market_info)
    if end_time is None:
        return None
    now_naive = now.astimezone(timezone.utc).replace(tzinfo=None) if now.tzinfo is not None else now
    return max(0.0, (end_time - now_naive).total_seconds())


def _market_end_time_iso(market_info: Any) -> Optional[str]:
    end_time = _extract_market_end_time_naive(market_info)
    if end_time is None:
        return None
    return _iso_utc(end_time)


async def _resolve_execution_wallet_address() -> str:
    wallet = ""
    try:
        await live_execution_service.ensure_initialized()
    except Exception:
        pass
    try:
        wallet = str(live_execution_service.get_execution_wallet_address() or "").strip()
    except Exception:
        wallet = ""
    if wallet:
        return wallet
    try:
        runtime_signature_type = getattr(live_execution_service, "_balance_signature_type", None)
        signature_type = (
            int(runtime_signature_type)
            if isinstance(runtime_signature_type, int)
            else int(getattr(settings, "POLYMARKET_SIGNATURE_TYPE", 1))
        )
    except Exception:
        signature_type = 1
    try:
        wallet = str(live_execution_service._funder_for_signature_type(signature_type) or "").strip()
    except Exception:
        wallet = ""
    if not wallet:
        wallet = str(live_execution_service._get_wallet_address() or "").strip()
    return wallet


async def _load_execution_wallet_positions_by_token() -> dict[str, dict[str, Any]]:
    wallet = await _resolve_execution_wallet_address()
    if not wallet:
        return {}

    try:
        positions = await polymarket_client.get_wallet_positions(wallet)
    except Exception:
        return {}

    by_token: dict[str, dict[str, Any]] = {}
    for position in positions:
        if not isinstance(position, dict):
            continue
        token_id = str(position.get("asset") or position.get("asset_id") or position.get("token_id") or "").strip()
        if not token_id:
            continue
        by_token[token_id] = position
    return by_token


async def _load_execution_wallet_recent_sell_trades_by_token() -> dict[str, dict[str, Any]]:
    wallet = await _resolve_execution_wallet_address()
    if not wallet:
        return {}
    try:
        trades = await polymarket_client.get_wallet_trades(wallet, limit=250)
    except Exception:
        return {}
    if not isinstance(trades, list):
        return {}

    latest_by_token: dict[str, dict[str, Any]] = {}
    market_cache_by_condition: dict[str, Optional[dict[str, Any]]] = {}

    async def _infer_trade_token_id(trade: dict[str, Any]) -> str:
        condition_id = str(trade.get("conditionId") or trade.get("condition_id") or trade.get("market") or "").strip()
        if not condition_id:
            return ""

        if condition_id not in market_cache_by_condition:
            market_cache_by_condition[condition_id] = await polymarket_client.get_market_by_condition_id(condition_id)
        market_info = market_cache_by_condition.get(condition_id)
        if not isinstance(market_info, dict):
            return ""

        token_ids_raw = market_info.get("token_ids")
        if not isinstance(token_ids_raw, list):
            token_ids_raw = market_info.get("tokenIds")
        if not isinstance(token_ids_raw, list):
            return ""
        token_ids = [str(token_id or "").strip() for token_id in token_ids_raw]
        if not token_ids:
            return ""

        outcomes_raw = market_info.get("outcomes")
        outcomes: list[str] = []
        if isinstance(outcomes_raw, list):
            outcomes = [str(outcome or "").strip().lower() for outcome in outcomes_raw]

        outcome_idx = safe_float(
            trade.get("outcomeIndex") if trade.get("outcomeIndex") is not None else trade.get("outcome_index")
        )
        if outcome_idx is not None:
            idx = int(outcome_idx)
            if 0 <= idx < len(token_ids):
                return token_ids[idx]

        outcome_text = str(trade.get("outcome") or trade.get("token_outcome") or "").strip().lower()
        if outcome_text and outcomes:
            for idx, outcome_label in enumerate(outcomes):
                if outcome_label == outcome_text and idx < len(token_ids):
                    return token_ids[idx]

        return ""

    for trade in trades:
        if not isinstance(trade, dict):
            continue
        token_id = _extract_wallet_trade_token_id(trade)
        if not token_id:
            token_id = await _infer_trade_token_id(trade)
        if not token_id:
            continue
        if _extract_wallet_trade_side(trade) != "sell":
            continue
        size = _extract_wallet_trade_size(trade)
        if size <= 0.0:
            continue
        price = _extract_wallet_trade_price(trade)
        timestamp = _parse_wallet_trade_time(
            trade.get("timestamp") or trade.get("created_at") or trade.get("createdAt") or trade.get("time")
        )
        record = {
            "trade_id": str(trade.get("id") or trade.get("order_id") or trade.get("orderId") or "").strip(),
            "token_id": token_id,
            "size": size,
            "price": price,
            "timestamp": timestamp,
        }
        existing = latest_by_token.get(token_id)
        if existing is None:
            latest_by_token[token_id] = record
            continue
        existing_ts = existing.get("timestamp")
        if timestamp is not None and (
            existing_ts is None or (isinstance(existing_ts, datetime) and timestamp > existing_ts)
        ):
            latest_by_token[token_id] = record
    return latest_by_token


async def load_market_info_for_orders(orders: list[TraderOrder]) -> dict[str, Optional[dict[str, Any]]]:
    market_ids = sorted({str(order.market_id or "").strip() for order in orders if str(order.market_id or "").strip()})
    if not market_ids:
        return {}

    async def _fetch(market_id: str) -> tuple[str, Optional[dict[str, Any]]]:
        info: Optional[dict[str, Any]] = None
        if market_id.startswith("0x"):
            # Lifecycle decisions must use fresh market metadata so terminal
            # resolution state (closed/winner/outcome prices) is not blocked
            # by stale in-memory cache entries in long-lived workers.
            info = await polymarket_client.get_market_by_condition_id(market_id, force_refresh=True)
            if info is None:
                info = await polymarket_client.get_market_by_condition_id(market_id)
        if info is None:
            info = await polymarket_client.get_market_by_token_id(market_id, force_refresh=True)
            if info is None:
                info = await polymarket_client.get_market_by_token_id(market_id)
        return market_id, info

    pairs = await asyncio.gather(*[_fetch(market_id) for market_id in market_ids], return_exceptions=True)
    out: dict[str, Optional[dict[str, Any]]] = {}
    for item in pairs:
        if isinstance(item, Exception):
            continue
        market_id, info = item
        out[market_id] = info
    return out


async def reconcile_paper_positions(
    session: AsyncSession,
    *,
    trader_id: str,
    trader_params: Optional[dict[str, Any]] = None,
    dry_run: bool = False,
    force_mark_to_market: bool = False,
    max_age_hours: Optional[int] = None,
    reason: str = "paper_position_lifecycle",
) -> dict[str, Any]:
    params = dict(trader_params or {})
    take_profit_pct = safe_float(params.get("paper_take_profit_pct"))
    stop_loss_pct = safe_float(params.get("paper_stop_loss_pct"))
    max_hold_minutes = safe_float(params.get("paper_max_hold_minutes"))
    min_hold_minutes = max(0.0, safe_float(params.get("paper_min_hold_minutes")) or 0.0)
    trailing_stop_pct = safe_float(params.get("paper_trailing_stop_pct"))
    resolve_only = _safe_bool(params.get("paper_resolve_only"), False)
    close_on_inactive_market = _safe_bool(params.get("paper_close_on_inactive_market"), False)
    resolution_infer_from_prices = _safe_bool(params.get("paper_resolution_infer_from_prices"), True)
    resolution_settle_floor = min(
        1.0,
        max(0.5, safe_float(params.get("paper_resolution_settle_floor")) or 0.98),
    )

    candidates = list(
        (
            await session.execute(
                select(TraderOrder).where(
                    TraderOrder.trader_id == trader_id,
                    func.lower(func.coalesce(TraderOrder.mode, "")) == "paper",
                    func.lower(func.coalesce(TraderOrder.status, "")).in_(tuple(PAPER_ACTIVE_STATUSES)),
                )
            )
        )
        .scalars()
        .all()
    )

    if max_age_hours is not None:
        cutoff = utcnow() - timedelta(hours=max(1, int(max_age_hours)))
        candidates = [
            row
            for row in candidates
            if (row.executed_at or row.updated_at or row.created_at) is not None
            and (row.executed_at or row.updated_at or row.created_at) <= cutoff
        ]

    signal_ids = [str(row.signal_id) for row in candidates if row.signal_id]
    signal_payloads: dict[str, dict[str, Any]] = {}
    if signal_ids:
        signal_rows = (
            await session.execute(
                select(TradeSignal.id, TradeSignal.payload_json).where(TradeSignal.id.in_(signal_ids))
            )
        ).all()
        signal_payloads = {str(row.id): dict(row.payload_json or {}) for row in signal_rows}

    market_info_by_id = await load_market_info_for_orders(candidates)

    now = utcnow()
    would_close = 0
    closed = 0
    held = 0
    skipped = 0
    total_realized_pnl = 0.0
    by_status = {"resolved_win": 0, "resolved_loss": 0, "closed_win": 0, "closed_loss": 0}
    skipped_reasons: dict[str, int] = {}
    details: list[dict[str, Any]] = []
    state_updates = 0
    reverse_signal_ids_by_source: dict[str, list[str]] = {}

    for row in candidates:
        entry_price = safe_float(row.effective_price)
        if entry_price is None or entry_price <= 0:
            entry_price = safe_float(row.entry_price)
        notional = safe_float(row.notional_usd) or 0.0
        outcome_idx = _direction_outcome_index(row.direction)
        if outcome_idx is None or entry_price is None or entry_price <= 0 or notional <= 0:
            skipped += 1
            skipped_reasons["invalid_entry"] = int(skipped_reasons.get("invalid_entry", 0)) + 1
            continue

        signal_payload = signal_payloads.get(str(row.signal_id), {})
        market_info = market_info_by_id.get(str(row.market_id or ""))
        market_tradable = polymarket_client.is_market_tradable(market_info, now=now)
        market_seconds_left = _market_seconds_left(market_info, now)
        market_end_time = _market_end_time_iso(market_info)
        winning_idx = _extract_winning_outcome_index(market_info)
        winning_idx_inferred = False
        if winning_idx is None and resolution_infer_from_prices:
            inferred_idx = _extract_winning_outcome_index_from_prices(
                market_info,
                market_tradable=market_tradable,
                settle_floor=resolution_settle_floor,
            )
            if inferred_idx is not None:
                winning_idx = inferred_idx
                winning_idx_inferred = True
        market_side_price = _extract_market_side_price(market_info, outcome_idx)
        snapshot_side_price = _extract_signal_side_price(signal_payload, outcome_idx)

        close_price: Optional[float] = None
        close_trigger: Optional[str] = None
        price_source: Optional[str] = None
        trailing_trigger_price: Optional[float] = None

        current_price = market_side_price if market_side_price is not None else snapshot_side_price
        current_price = _state_price_floor(current_price)
        current_price_source = (
            "market_mark"
            if market_side_price is not None
            else ("signal_snapshot_mark" if snapshot_side_price is not None else None)
        )

        age_anchor = row.executed_at or row.updated_at or row.created_at
        age_minutes = None
        if age_anchor is not None:
            age_minutes = max(0.0, (now - age_anchor).total_seconds() / 60.0)
        min_hold_passed = age_minutes is None or age_minutes >= min_hold_minutes

        payload = dict(row.payload_json or {})
        position_state = _extract_position_state(payload)
        prev_high = safe_float(position_state.get("highest_price"))
        prev_low = safe_float(position_state.get("lowest_price"))
        prev_last_mark = safe_float(position_state.get("last_mark_price"))
        prev_mark_source = str(position_state.get("last_mark_source") or "")
        highest_price = prev_high
        lowest_price = prev_low
        if current_price is not None:
            if highest_price is None:
                highest_price = current_price
            else:
                highest_price = max(highest_price, current_price)
            if lowest_price is None:
                lowest_price = current_price
            else:
                lowest_price = min(lowest_price, current_price)

        next_state = {
            "highest_price": _state_price_floor(highest_price),
            "lowest_price": _state_price_floor(lowest_price),
            "last_mark_price": current_price,
            "last_mark_source": current_price_source,
            "last_marked_at": _iso_utc(now),
        }

        if winning_idx is not None:
            close_price = 1.0 if winning_idx == outcome_idx else 0.0
            close_trigger = "resolution_inferred" if winning_idx_inferred else "resolution"
            price_source = "resolved_settlement"
        else:
            pnl_pct = None
            if current_price is not None and entry_price > 0:
                pnl_pct = ((current_price - entry_price) / entry_price) * 100.0

            if force_mark_to_market and current_price is not None:
                close_price = current_price
                close_trigger = "manual_mark_to_market"
                price_source = current_price_source
            else:
                # ── Strategy-based exit check ──────────────────────────
                # If the strategy that opened this position has a
                # should_exit() method, call it first and respect its
                # decision before falling through to default TP/SL/etc.
                strategy_slug = (payload.get("strategy_type") or "").strip().lower()
                strategy_exit = None
                _exit_instance = None
                if strategy_slug:
                    from services.strategy_loader import strategy_loader

                    loaded = strategy_loader.get_strategy(strategy_slug)
                    if loaded and hasattr(loaded.instance, "should_exit"):
                        _exit_instance = loaded.instance
                if _exit_instance is not None:
                    try:

                        class _PaperPositionView:
                            pass

                        pos_view = _PaperPositionView()
                        pos_view.entry_price = entry_price
                        pos_view.current_price = current_price
                        pos_view.highest_price = highest_price
                        pos_view.lowest_price = lowest_price
                        pos_view.age_minutes = age_minutes
                        pos_view.pnl_percent = pnl_pct
                        pos_view.filled_size = (notional / entry_price) if entry_price > 0 else 0.0
                        pos_view.notional_usd = notional
                        if "strategy_context" not in payload:
                            payload["strategy_context"] = {}
                        pos_view.strategy_context = payload["strategy_context"]
                        pos_view.config = payload.get("strategy_exit_config", {})
                        pos_view.outcome_idx = outcome_idx

                        token_id = _extract_live_token_id(payload)
                        min_order_size_usd = _resolve_position_min_order_size_usd(
                            trader_params=params,
                            payload=payload,
                            mode="paper",
                        )
                        market_state_dict = {
                            "current_price": current_price,
                            "market_tradable": market_tradable,
                            "is_resolved": False,
                            "winning_outcome": None,
                            "seconds_left": market_seconds_left,
                            "end_time": market_end_time,
                            "token_id": token_id,
                            "mark_source": current_price_source,
                            "min_order_size_usd": min_order_size_usd,
                            "notional_usd": notional,
                        }

                        exit_decision = _exit_instance.should_exit(pos_view, market_state_dict)
                        exit_action = getattr(exit_decision, "action", None) if exit_decision is not None else None
                        if exit_action == "close":
                            strategy_exit = exit_decision
                        elif exit_action == "reduce":
                            strategy_exit = exit_decision
                    except Exception as exc:
                        logger.warning(
                            "Strategy should_exit() error for %s: %s",
                            strategy_slug,
                            exc,
                        )

                if strategy_exit is not None and getattr(strategy_exit, "action", None) == "reduce":
                    if not dry_run:
                        payload["position_state"] = next_state
                        row.payload_json = payload
                        row.updated_at = now
                        state_updates += 1
                    held += 1
                    continue

                if strategy_exit is not None:
                    _arm_reverse_entry_from_exit(
                        row=row,
                        payload=payload,
                        strategy_exit=strategy_exit,
                        market_info=market_info,
                        market_seconds_left=market_seconds_left,
                        current_price=current_price,
                        now=now,
                    )
                    close_price = strategy_exit.close_price if strategy_exit.close_price is not None else current_price
                    close_trigger = f"strategy:{strategy_exit.reason}"
                    price_source = current_price_source
                elif (
                    not resolve_only
                    and take_profit_pct is not None
                    and pnl_pct is not None
                    and min_hold_passed
                    and pnl_pct >= take_profit_pct
                ):
                    close_price = current_price
                    close_trigger = "take_profit"
                    price_source = current_price_source
                elif (
                    not resolve_only
                    and stop_loss_pct is not None
                    and pnl_pct is not None
                    and min_hold_passed
                    and pnl_pct <= -abs(stop_loss_pct)
                ):
                    close_price = current_price
                    close_trigger = "stop_loss"
                    price_source = current_price_source
                elif (
                    not resolve_only
                    and trailing_stop_pct is not None
                    and trailing_stop_pct > 0
                    and current_price is not None
                    and highest_price is not None
                    and min_hold_passed
                ):
                    trailing_trigger_price = highest_price * (1.0 - (trailing_stop_pct / 100.0))
                    if highest_price > entry_price and current_price <= trailing_trigger_price:
                        close_price = current_price
                        close_trigger = "trailing_stop"
                        price_source = current_price_source
                elif (
                    not resolve_only
                    and max_hold_minutes is not None
                    and age_minutes is not None
                    and age_minutes >= max_hold_minutes
                ):
                    if current_price is not None:
                        close_price = current_price
                        close_trigger = "max_hold"
                        price_source = current_price_source
                elif (
                    not resolve_only
                    and close_on_inactive_market
                    and not market_tradable
                    and current_price is not None
                    and min_hold_passed
                ):
                    close_price = current_price
                    close_trigger = "market_inactive"
                    price_source = current_price_source

        if close_price is None:
            state_changed = False
            if current_price is not None:
                state_changed = (
                    prev_last_mark is None
                    or abs(prev_last_mark - current_price) > 1e-9
                    or prev_high is None
                    or prev_low is None
                    or abs((prev_high or 0.0) - (highest_price or 0.0)) > 1e-9
                    or abs((prev_low or 0.0) - (lowest_price or 0.0)) > 1e-9
                    or prev_mark_source != str(current_price_source or "")
                )
            if not dry_run and state_changed:
                payload["position_state"] = next_state
                row.payload_json = payload
                row.updated_at = now
                state_updates += 1
            held += 1
            continue

        quantity = notional / entry_price
        proceeds = quantity * close_price
        pnl = proceeds - notional
        next_status = _status_for_close(pnl=pnl, close_trigger=close_trigger)

        simulation_close: dict[str, Any] | None = None
        simulation_ledger = payload.get("simulation_ledger")
        if not dry_run and isinstance(simulation_ledger, dict):
            sim_account_id = str(simulation_ledger.get("account_id") or "").strip()
            sim_trade_id = str(simulation_ledger.get("trade_id") or "").strip()
            sim_position_id = str(simulation_ledger.get("position_id") or "").strip()
            if sim_account_id and sim_trade_id and sim_position_id:
                try:
                    simulation_close = await simulation_service.close_orchestrator_paper_fill(
                        account_id=sim_account_id,
                        trade_id=sim_trade_id,
                        position_id=sim_position_id,
                        close_price=float(close_price),
                        close_trigger=close_trigger,
                        price_source=price_source,
                        reason=reason,
                        session=session,
                        commit=False,
                    )
                    if simulation_close.get("closed"):
                        proceeds = float(simulation_close.get("actual_payout", proceeds))
                        pnl = float(simulation_close.get("actual_pnl", pnl))
                        next_status = str(simulation_close.get("trade_status") or next_status)
                    elif simulation_close.get("already_closed"):
                        existing_status = str(simulation_close.get("trade_status") or "")
                        if existing_status:
                            next_status = existing_status
                        pnl = float(simulation_close.get("actual_pnl", pnl))
                        proceeds = float(simulation_close.get("actual_payout", proceeds))
                except Exception as exc:
                    skipped += 1
                    skipped_reasons["simulation_close_error"] = (
                        int(skipped_reasons.get("simulation_close_error", 0)) + 1
                    )
                    details.append(
                        {
                            "order_id": row.id,
                            "market_id": row.market_id,
                            "direction": row.direction,
                            "close_trigger": close_trigger,
                            "reason": "simulation_close_error",
                            "error": str(exc),
                        }
                    )
                    continue

        total_realized_pnl += pnl
        by_status[next_status] = int(by_status.get(next_status, 0)) + 1

        detail = {
            "order_id": row.id,
            "market_id": row.market_id,
            "direction": row.direction,
            "entry_price": entry_price,
            "close_price": close_price,
            "price_source": price_source,
            "close_trigger": close_trigger,
            "market_tradable": market_tradable,
            "notional_usd": notional,
            "quantity": quantity,
            "realized_pnl": pnl,
            "next_status": next_status,
            "age_minutes": age_minutes,
            "min_hold_minutes": min_hold_minutes,
            "trailing_stop_trigger_price": trailing_trigger_price,
            "highest_price_seen": _state_price_floor(highest_price),
            "lowest_price_seen": _state_price_floor(lowest_price),
            "simulation_close": simulation_close,
        }
        details.append(detail)
        would_close += 1

        if dry_run:
            continue

        row.status = next_status
        row.actual_profit = pnl
        row.updated_at = now
        payload["position_state"] = next_state
        payload["position_close"] = {
            "close_price": close_price,
            "price_source": price_source,
            "close_trigger": close_trigger,
            "realized_pnl": pnl,
            "market_tradable": market_tradable,
            "age_minutes": age_minutes,
            "closed_at": _iso_utc(now),
            "reason": reason,
        }
        if simulation_close is not None:
            payload["position_close"]["simulation_close"] = simulation_close

        reverse_signal_id, reverse_source = await _emit_armed_reverse_signal(
            session,
            row=row,
            payload=payload,
            signal_payload=signal_payload,
            market_info=market_info,
            close_trigger=close_trigger,
            realized_pnl=pnl,
            now=now,
        )
        if reverse_signal_id and reverse_source:
            reverse_signal_ids_by_source.setdefault(reverse_source, []).append(reverse_signal_id)

        row.payload_json = payload
        if reason:
            if row.reason:
                row.reason = f"{row.reason} | {reason}:{close_trigger}"
            else:
                row.reason = f"{reason}:{close_trigger}"
        closed += 1

    if not dry_run and (closed > 0 or state_updates > 0):
        await session.commit()
        await _publish_trader_order_updates([row for row in candidates if row.updated_at == now])
        if reverse_signal_ids_by_source:
            await refresh_trade_signal_snapshots(session)
            await _publish_reverse_signal_batches(reverse_signal_ids_by_source, emitted_at=now)

    return {
        "trader_id": trader_id,
        "dry_run": bool(dry_run),
        "matched": len(candidates),
        "would_close": would_close,
        "closed": closed,
        "held": held,
        "skipped": skipped,
        "state_updates": state_updates,
        "total_realized_pnl": total_realized_pnl,
        "by_status": by_status,
        "skipped_reasons": skipped_reasons,
        "reverse_signals_emitted": sum(len(ids) for ids in reverse_signal_ids_by_source.values()),
        "details": details,
    }


async def reconcile_live_positions(
    session: AsyncSession,
    *,
    trader_id: str,
    trader_params: Optional[dict[str, Any]] = None,
    dry_run: bool = False,
    force_mark_to_market: bool = False,
    max_age_hours: Optional[int] = None,
    reason: str = "live_position_lifecycle",
) -> dict[str, Any]:
    """Lifecycle management for live positions.

    Mirrors reconcile_paper_positions but operates on mode='live' orders.
    Handles: stop-loss, take-profit, trailing stop, max hold, market
    inactivity, and resolution detection.  Does NOT interact with the
    simulation ledger (that is paper-only).
    """
    params = dict(trader_params or {})
    take_profit_pct = safe_float(params.get("live_take_profit_pct"))
    stop_loss_pct = safe_float(params.get("live_stop_loss_pct"))
    max_hold_minutes = safe_float(params.get("live_max_hold_minutes"))
    min_hold_minutes = max(0.0, safe_float(params.get("live_min_hold_minutes")) or 0.0)
    trailing_stop_pct = safe_float(params.get("live_trailing_stop_pct"))
    resolve_only = _safe_bool(params.get("live_resolve_only"), False)
    close_on_inactive_market = _safe_bool(params.get("live_close_on_inactive_market"), False)
    resolution_infer_from_prices = _safe_bool(params.get("live_resolution_infer_from_prices"), True)
    resolution_settle_floor = min(
        1.0,
        max(0.5, safe_float(params.get("live_resolution_settle_floor")) or 0.98),
    )
    mark_touch_interval_seconds = _mark_touch_interval_seconds(params, mode="live")

    candidates = list(
        (
            await session.execute(
                select(TraderOrder).where(
                    TraderOrder.trader_id == trader_id,
                    func.lower(func.coalesce(TraderOrder.mode, "")) == "live",
                    func.lower(func.coalesce(TraderOrder.status, "")).in_(tuple(LIVE_ACTIVE_STATUSES)),
                )
            )
        )
        .scalars()
        .all()
    )

    if max_age_hours is not None:
        cutoff = utcnow() - timedelta(hours=max(1, int(max_age_hours)))
        candidates = [
            row
            for row in candidates
            if (row.executed_at or row.updated_at or row.created_at) is not None
            and (row.executed_at or row.updated_at or row.created_at) <= cutoff
        ]

    now = utcnow()
    now_naive = now.astimezone(timezone.utc).replace(tzinfo=None) if now.tzinfo is not None else now
    would_close = 0
    closed = 0
    held = 0
    skipped = 0
    total_realized_pnl = 0.0
    by_status: dict[str, int] = {"resolved_win": 0, "resolved_loss": 0, "closed_win": 0, "closed_loss": 0}
    skipped_reasons: dict[str, int] = {}
    details: list[dict[str, Any]] = []
    state_updates = 0
    reverse_signal_ids_by_source: dict[str, list[str]] = {}

    candidate_ids = {str(row.id) for row in candidates}
    terminal_rows = list(
        (
            await session.execute(
                select(TraderOrder).where(
                    TraderOrder.trader_id == trader_id,
                    func.lower(func.coalesce(TraderOrder.mode, "")) == "live",
                    func.lower(func.coalesce(TraderOrder.status, "")).in_(("closed_win", "closed_loss")),
                )
            )
        )
        .scalars()
        .all()
    )
    if max_age_hours is not None:
        cutoff = now - timedelta(hours=max(1, int(max_age_hours)))
        terminal_rows = [
            row
            for row in terminal_rows
            if (row.executed_at or row.updated_at or row.created_at) is not None
            and (row.executed_at or row.updated_at or row.created_at) <= cutoff
        ]

    for row in terminal_rows:
        row_id = str(row.id)
        if row_id in candidate_ids:
            continue
        payload = dict(row.payload_json or {})
        pending_exit = payload.get("pending_live_exit")
        if not isinstance(pending_exit, dict):
            continue
        pending_status = str(pending_exit.get("status") or "").strip().lower()
        if pending_status != "filled":
            continue
        if _safe_bool(pending_exit.get("allow_partial_fill_terminal"), False):
            continue
        close_trigger = str(pending_exit.get("close_trigger") or "").strip().lower()
        if "resolution" in close_trigger:
            continue
        required_exit_size = max(0.0, safe_float(pending_exit.get("exit_size"), 0.0) or 0.0)
        _entry_notional, _entry_size, _entry_price = _extract_live_fill_metrics(payload)
        if _entry_size <= 0.0 and _entry_notional > 0.0:
            _fallback_entry_price = (
                _entry_price if _entry_price and _entry_price > 0 else safe_float(row.effective_price)
            )
            if _fallback_entry_price is None or _fallback_entry_price <= 0:
                _fallback_entry_price = safe_float(row.entry_price)
            if _fallback_entry_price and _fallback_entry_price > 0:
                _entry_size = _entry_notional / _fallback_entry_price
        if _entry_size > 0.0:
            required_exit_size = max(required_exit_size, _entry_size)
        if required_exit_size <= 0.0:
            continue
        filled_exit_size = max(0.0, safe_float(pending_exit.get("filled_size"), 0.0) or 0.0)
        fill_ratio = safe_float(pending_exit.get("fill_ratio"))
        if fill_ratio is None:
            fill_ratio = (filled_exit_size / required_exit_size) if required_exit_size > 0.0 else 0.0
        threshold_ratio = _pending_exit_fill_threshold(pending_exit)
        if fill_ratio + 1e-9 >= threshold_ratio:
            continue
        if not dry_run:
            row.status = "open"
            row.actual_profit = None
            row.updated_at = now
            pending_exit["status"] = (
                "submitted"
                if str(pending_exit.get("exit_order_id") or "").strip()
                or str(pending_exit.get("provider_clob_order_id") or "").strip()
                else "pending"
            )
            pending_exit["reopened_at"] = _iso_utc(now)
            pending_exit["reopen_reason"] = "partial_exit_fill_below_threshold"
            pending_exit["fill_ratio"] = float(fill_ratio)
            payload["pending_live_exit"] = pending_exit
            payload.pop("position_close", None)
            row.payload_json = payload
            state_updates += 1
        details.append(
            {
                "order_id": row.id,
                "market_id": row.market_id,
                "close_trigger": str(pending_exit.get("close_trigger") or "live_exit_fill"),
                "filled_size": filled_exit_size,
                "required_exit_size": required_exit_size,
                "fill_ratio": float(fill_ratio),
                "fill_ratio_threshold": threshold_ratio,
                "next_status": "open",
                "note": "reopened_partial_exit_fill_below_threshold",
            }
        )
        candidates.append(row)
        candidate_ids.add(row_id)

    signal_ids = [str(row.signal_id) for row in candidates if row.signal_id]
    signal_payloads: dict[str, dict[str, Any]] = {}
    if signal_ids:
        signal_rows = (
            await session.execute(
                select(TradeSignal.id, TradeSignal.payload_json).where(TradeSignal.id.in_(signal_ids))
            )
        ).all()
        signal_payloads = {str(row.id): dict(row.payload_json or {}) for row in signal_rows}

    market_info_by_id = await load_market_info_for_orders(candidates)
    wallet_positions_by_token = await _load_execution_wallet_positions_by_token()
    wallet_sell_trades_by_token = await _load_execution_wallet_recent_sell_trades_by_token()
    redis_mid_prices: dict[str, float] = {}
    clob_mid_prices: dict[str, float] = {}
    token_ids_for_prices = sorted(
        {
            token_id
            for token_id in (_extract_live_token_id(dict(row.payload_json or {})) for row in candidates)
            if token_id
        }
    )
    if token_ids_for_prices:
        strict_stale_seconds = max(0.05, float(settings.WS_EXECUTION_PRICE_STALE_SECONDS))
        relaxed_stale_seconds = max(strict_stale_seconds, float(settings.WS_PRICE_STALE_SECONDS))
        try:
            from services.redis_price_cache import redis_price_cache

            redis_rows = await redis_price_cache.read_prices(
                token_ids_for_prices,
                stale_seconds=strict_stale_seconds,
            )
            for token_id, price_row in redis_rows.items():
                mid = safe_float((price_row or {}).get("mid")) if isinstance(price_row, dict) else None
                if mid is not None and mid >= 0:
                    redis_mid_prices[str(token_id).strip()] = float(mid)
        except Exception:
            redis_mid_prices = {}
        unresolved_token_ids = [token_id for token_id in token_ids_for_prices if token_id not in redis_mid_prices]
        if unresolved_token_ids and relaxed_stale_seconds > strict_stale_seconds + 1e-9:
            try:
                from services.redis_price_cache import redis_price_cache

                relaxed_rows = await redis_price_cache.read_prices(
                    unresolved_token_ids,
                    stale_seconds=relaxed_stale_seconds,
                )
                for token_id, price_row in relaxed_rows.items():
                    mid = safe_float((price_row or {}).get("mid")) if isinstance(price_row, dict) else None
                    if mid is not None and mid >= 0:
                        redis_mid_prices[str(token_id).strip()] = float(mid)
            except Exception:
                pass
        unresolved_token_ids = [token_id for token_id in token_ids_for_prices if token_id not in redis_mid_prices]
        if unresolved_token_ids:

            async def _fetch_clob_midpoint(token_id: str) -> tuple[str, Optional[float]]:
                try:
                    midpoint = await asyncio.wait_for(
                        polymarket_client.get_midpoint(token_id),
                        timeout=1.5,
                    )
                except Exception:
                    return token_id, None
                parsed_midpoint = safe_float(midpoint)
                if parsed_midpoint is None or parsed_midpoint < 0:
                    return token_id, None
                return token_id, float(parsed_midpoint)

            midpoint_pairs = await asyncio.gather(
                *[_fetch_clob_midpoint(token_id) for token_id in unresolved_token_ids],
                return_exceptions=True,
            )
            for item in midpoint_pairs:
                if isinstance(item, Exception):
                    continue
                token_id, midpoint = item
                if midpoint is not None:
                    clob_mid_prices[str(token_id).strip()] = midpoint

    pending_exit_provider_ids = sorted(
        {
            _pending_exit_provider_clob_id(pending_exit)
            for pending_exit in ((dict((row.payload_json or {})).get("pending_live_exit")) for row in candidates)
            if isinstance(pending_exit, dict)
            and str(pending_exit.get("status") or "").strip().lower() in {"submitted", "pending"}
            and _pending_exit_provider_clob_id(pending_exit)
        }
    )
    pending_exit_snapshots: dict[str, dict[str, Any]] = {}
    if pending_exit_provider_ids:
        try:
            pending_exit_snapshots = await live_execution_service.get_order_snapshots_by_clob_ids(pending_exit_provider_ids)
        except Exception:
            pending_exit_snapshots = {}

    for row in candidates:
        payload = dict(row.payload_json or {})
        token_id = _extract_live_token_id(payload)
        wallet_position = wallet_positions_by_token.get(token_id) if token_id else None
        wallet_position_observed = isinstance(wallet_position, dict)
        wallet_position_size = _extract_wallet_position_size(wallet_position)
        wallet_mark_price = _extract_wallet_mark_price(wallet_position)
        wallet_settlement_price = _extract_wallet_settlement_price(wallet_position)
        latest_wallet_sell_trade = wallet_sell_trades_by_token.get(token_id) if token_id else None
        entry_fill_notional, entry_fill_size, entry_fill_price = _extract_live_fill_metrics(payload)
        if entry_fill_price is None or entry_fill_price <= 0.0:
            entry_fill_price = safe_float(row.effective_price)
        if entry_fill_price is None or entry_fill_price <= 0.0:
            entry_fill_price = safe_float(row.entry_price)
        if entry_fill_notional <= 0.0:
            entry_fill_notional = max(0.0, safe_float(row.notional_usd) or 0.0)
        if entry_fill_size <= 0.0 and entry_fill_notional > 0.0 and entry_fill_price and entry_fill_price > 0.0:
            entry_fill_size = entry_fill_notional / entry_fill_price
        pending_outcome_idx = _direction_outcome_index(row.direction)
        pending_market_info = market_info_by_id.get(str(row.market_id or ""))
        pending_market_tradable = polymarket_client.is_market_tradable(pending_market_info, now=now)
        pending_winning_idx = _extract_winning_outcome_index(pending_market_info)
        pending_winning_idx_inferred = False
        if pending_winning_idx is None and resolution_infer_from_prices:
            inferred_pending_idx = _extract_winning_outcome_index_from_prices(
                pending_market_info,
                market_tradable=pending_market_tradable,
                settle_floor=resolution_settle_floor,
            )
            if inferred_pending_idx is not None:
                pending_winning_idx = inferred_pending_idx
                pending_winning_idx_inferred = True

        pending_signal_payload = signal_payloads.get(str(row.signal_id), {})
        pending_market_side_price = (
            _extract_market_side_price(pending_market_info, pending_outcome_idx)
            if pending_outcome_idx is not None
            else None
        )
        pending_redis_side_price = redis_mid_prices.get(token_id) if token_id else None
        pending_clob_side_price = clob_mid_prices.get(token_id) if token_id else None
        pending_current_price = (
            pending_redis_side_price
            if pending_redis_side_price is not None
            else (
                pending_clob_side_price
                if pending_clob_side_price is not None
                else (
                    pending_market_side_price
                    if pending_market_side_price is not None
                    else wallet_mark_price
                )
            )
        )
        pending_current_price = _state_price_floor(pending_current_price)
        pending_current_price_source = (
            "redis_mid"
            if pending_redis_side_price is not None
            else (
                "clob_midpoint"
                if pending_clob_side_price is not None
                else (
                    "market_mark"
                    if pending_market_side_price is not None
                    else ("wallet_mark" if wallet_mark_price is not None else None)
                )
            )
        )
        pending_position_state = _extract_position_state(payload)
        pending_prev_high = safe_float(pending_position_state.get("highest_price"))
        pending_prev_low = safe_float(pending_position_state.get("lowest_price"))
        pending_prev_last_mark = safe_float(pending_position_state.get("last_mark_price"))
        pending_prev_mark_source = str(pending_position_state.get("last_mark_source") or "")
        pending_prev_marked_at = _parse_iso_utc_naive(pending_position_state.get("last_marked_at"))
        pending_highest_price = pending_prev_high
        pending_lowest_price = pending_prev_low
        if pending_current_price is not None:
            if pending_highest_price is None:
                pending_highest_price = pending_current_price
            else:
                pending_highest_price = max(pending_highest_price, pending_current_price)
            if pending_lowest_price is None:
                pending_lowest_price = pending_current_price
            else:
                pending_lowest_price = min(pending_lowest_price, pending_current_price)
        pending_next_state = {
            "highest_price": _state_price_floor(pending_highest_price),
            "lowest_price": _state_price_floor(pending_lowest_price),
            "last_mark_price": pending_current_price,
            "last_mark_source": pending_current_price_source,
            "last_marked_at": _iso_utc(now),
        }
        pending_state_changed = False
        if pending_current_price is not None:
            pending_mark_stale = (
                pending_prev_marked_at is None
                or (now_naive - pending_prev_marked_at).total_seconds() >= mark_touch_interval_seconds
            )
            pending_state_changed = (
                pending_prev_last_mark is None
                or abs(pending_prev_last_mark - pending_current_price) > 1e-9
                or pending_prev_high is None
                or pending_prev_low is None
                or abs((pending_prev_high or 0.0) - (pending_highest_price or 0.0)) > 1e-9
                or abs((pending_prev_low or 0.0) - (pending_lowest_price or 0.0)) > 1e-9
                or pending_prev_mark_source != str(pending_current_price_source or "")
                or pending_mark_stale
            )

        # External/manual flatten convergence:
        # if wallet inventory is flat and we have concrete wallet SELL evidence
        # for this token after entry, treat the position as externally closed.
        if (
            token_id
            and entry_fill_size > 0.0
            and wallet_position_size <= _WALLET_SIZE_EPSILON
            and pending_winning_idx is None
            and wallet_settlement_price is None
            and isinstance(latest_wallet_sell_trade, dict)
        ):
            trade_ts = latest_wallet_sell_trade.get("timestamp")
            row_created = row.created_at
            trade_after_entry = True
            if isinstance(trade_ts, datetime) and isinstance(row_created, datetime):
                entry_anchor = row_created
                if entry_anchor.tzinfo is not None:
                    entry_anchor = entry_anchor.astimezone(timezone.utc).replace(tzinfo=None)
                trade_after_entry = trade_ts >= entry_anchor
            if trade_after_entry:
                close_price = safe_float(latest_wallet_sell_trade.get("price"))
                if close_price is None or close_price <= 0.0:
                    close_price = (
                        wallet_mark_price
                        if wallet_mark_price is not None and wallet_mark_price > 0.0
                        else entry_fill_price
                    )
                if close_price is not None and close_price > 0.0:
                    close_qty = entry_fill_size
                    close_notional = close_qty * close_price
                    close_trigger = "external_wallet_flatten"
                    pnl = close_notional - entry_fill_notional
                    next_status = _status_for_close(pnl=pnl, close_trigger=close_trigger)
                    details.append(
                        {
                            "order_id": row.id,
                            "market_id": row.market_id,
                            "close_trigger": close_trigger,
                            "close_price": close_price,
                            "realized_pnl": pnl,
                            "next_status": next_status,
                            "wallet_trade_id": latest_wallet_sell_trade.get("trade_id"),
                        }
                    )
                    would_close += 1
                    total_realized_pnl += pnl
                    by_status[next_status] = int(by_status.get(next_status, 0)) + 1
                    if not dry_run:
                        pending_exit_local = payload.get("pending_live_exit")
                        if isinstance(pending_exit_local, dict):
                            pending_exit_local["status"] = "superseded_external"
                            pending_exit_local["resolved_at"] = _iso_utc(now)
                            payload["pending_live_exit"] = pending_exit_local
                        row.status = next_status
                        row.actual_profit = pnl
                        row.updated_at = now
                        payload["position_close"] = {
                            "close_price": close_price,
                            "price_source": "wallet_trade",
                            "close_trigger": close_trigger,
                            "realized_pnl": pnl,
                            "filled_size": close_qty,
                            "closed_at": _iso_utc(now),
                            "reason": reason,
                            "wallet_trade_id": str(latest_wallet_sell_trade.get("trade_id") or ""),
                            "wallet_trade_timestamp": (
                                _iso_utc(latest_wallet_sell_trade.get("timestamp"))
                                if isinstance(latest_wallet_sell_trade.get("timestamp"), datetime)
                                else None
                            ),
                        }
                        reverse_signal_id, reverse_source = await _emit_armed_reverse_signal(
                            session,
                            row=row,
                            payload=payload,
                            signal_payload=pending_signal_payload,
                            market_info=pending_market_info,
                            close_trigger=close_trigger,
                            realized_pnl=pnl,
                            now=now,
                        )
                        if reverse_signal_id and reverse_source:
                            reverse_signal_ids_by_source.setdefault(reverse_source, []).append(reverse_signal_id)
                        row.payload_json = payload
                        closed += 1
                    continue

        pending_exit = payload.get("pending_live_exit")
        pending_exit_status = (
            str(pending_exit.get("status") or "").strip().lower() if isinstance(pending_exit, dict) else ""
        )
        pending_exit_kind = (
            str(pending_exit.get("kind") or "").strip().lower() if isinstance(pending_exit, dict) else ""
        )

        def _attach_pending_state(target_payload: dict[str, Any]) -> None:
            if pending_state_changed:
                target_payload["position_state"] = pending_next_state

        # ── Submitted/pending exit reconciliation ──────────────────────
        # For live exits we trust provider fill truth; a terminal local close
        # only happens once provider fill size reaches the required threshold.
        if isinstance(pending_exit, dict) and pending_exit_status in {"submitted", "pending"}:
            if pending_winning_idx is not None and pending_outcome_idx is not None:
                _fill_not, _fill_sz, _fill_px = _extract_live_fill_metrics(payload)
                _ep = _fill_px if _fill_px and _fill_px > 0 else safe_float(row.effective_price)
                if _ep is None or _ep <= 0:
                    _ep = safe_float(row.entry_price)
                _not = _fill_not if _fill_not > 0 else (safe_float(row.notional_usd) or 0.0)
                _qty = _fill_sz if _fill_sz > 0 else (_not / _ep if _ep and _ep > 0 else 0.0)
                cp = 1.0 if pending_winning_idx == pending_outcome_idx else 0.0
                close_trigger = "resolution_inferred" if pending_winning_idx_inferred else "resolution"
                _pnl = (_qty * cp) - _not if _qty > 0 and _not > 0 else 0.0
                ns = _status_for_close(pnl=_pnl, close_trigger=close_trigger)
                if not dry_run:
                    row.status = ns
                    row.actual_profit = _pnl
                    row.updated_at = now
                    pending_exit["status"] = "superseded_resolution"
                    pending_exit["resolved_at"] = _iso_utc(now)
                    payload["pending_live_exit"] = pending_exit
                    payload["position_close"] = {
                        "close_price": cp,
                        "price_source": "resolved_settlement",
                        "close_trigger": close_trigger,
                        "realized_pnl": _pnl,
                        "closed_at": _iso_utc(now),
                        "reason": reason,
                    }
                    row.payload_json = payload
                    closed += 1
                total_realized_pnl += _pnl
                by_status[ns] = int(by_status.get(ns, 0)) + 1
                would_close += 1
                details.append(
                    {
                        "order_id": row.id,
                        "market_id": row.market_id,
                        "close_trigger": close_trigger,
                        "close_price": cp,
                        "realized_pnl": _pnl,
                        "next_status": ns,
                    }
                )
                continue

            if wallet_settlement_price is not None:
                _fill_not, _fill_sz, _fill_px = _extract_live_fill_metrics(payload)
                _ep = _fill_px if _fill_px and _fill_px > 0 else safe_float(row.effective_price)
                if _ep is None or _ep <= 0:
                    _ep = safe_float(row.entry_price)
                _not = _fill_not if _fill_not > 0 else (safe_float(row.notional_usd) or 0.0)
                _qty = _fill_sz if _fill_sz > 0 else (_not / _ep if _ep and _ep > 0 else 0.0)
                cp = wallet_settlement_price
                _pnl = (_qty * cp) - _not if _qty > 0 and _not > 0 else 0.0
                ns = _status_for_close(pnl=_pnl, close_trigger="resolution")
                if not dry_run:
                    row.status = ns
                    row.actual_profit = _pnl
                    row.updated_at = now
                    pending_exit["status"] = "superseded_resolution"
                    pending_exit["resolved_at"] = _iso_utc(now)
                    payload["pending_live_exit"] = pending_exit
                    payload["position_close"] = {
                        "close_price": cp,
                        "price_source": "wallet_redeemable_mark",
                        "close_trigger": "resolution",
                        "realized_pnl": _pnl,
                        "closed_at": _iso_utc(now),
                        "reason": reason,
                    }
                    row.payload_json = payload
                    closed += 1
                total_realized_pnl += _pnl
                by_status[ns] = int(by_status.get(ns, 0)) + 1
                would_close += 1
                details.append(
                    {
                        "order_id": row.id,
                        "market_id": row.market_id,
                        "close_trigger": "resolution",
                        "close_price": cp,
                        "realized_pnl": _pnl,
                        "next_status": ns,
                    }
                )
                continue

            provider_clob_order_id = _pending_exit_provider_clob_id(pending_exit)
            if provider_clob_order_id:
                pending_exit["provider_clob_order_id"] = provider_clob_order_id
            snapshot = pending_exit_snapshots.get(provider_clob_order_id) if provider_clob_order_id else None
            snapshot_status = str((snapshot or {}).get("normalized_status") or "").strip().lower()
            snapshot_filled_size = max(0.0, safe_float((snapshot or {}).get("filled_size"), 0.0) or 0.0)
            snapshot_fill_price = safe_float((snapshot or {}).get("average_fill_price"))
            if snapshot_fill_price is None or snapshot_fill_price <= 0:
                snapshot_fill_price = safe_float((snapshot or {}).get("limit_price"))
            if not snapshot_status:
                snapshot_status = str(pending_exit.get("provider_status") or "").strip().lower()
            if snapshot_filled_size <= 0.0:
                snapshot_filled_size = max(0.0, safe_float(pending_exit.get("filled_size"), 0.0) or 0.0)
            if snapshot_fill_price is None or snapshot_fill_price <= 0.0:
                snapshot_fill_price = safe_float(pending_exit.get("average_fill_price"))
            required_exit_size = max(0.0, safe_float(pending_exit.get("exit_size"), 0.0) or 0.0)
            _fill_not, _fill_sz, _fill_px = _extract_live_fill_metrics(payload)
            _ep = _fill_px if _fill_px and _fill_px > 0 else safe_float(row.effective_price)
            if _ep is None or _ep <= 0:
                _ep = safe_float(row.entry_price)
            _not = _fill_not if _fill_not > 0 else (safe_float(row.notional_usd) or 0.0)
            entry_position_size = _fill_sz if _fill_sz > 0 else (_not / _ep if _ep and _ep > 0 else 0.0)
            if entry_position_size > 0.0:
                required_exit_size = max(required_exit_size, entry_position_size)
            if required_exit_size > 0.0:
                pending_exit["exit_size"] = float(required_exit_size)

            threshold_ratio = _pending_exit_fill_threshold(pending_exit)
            fill_ratio = (
                (snapshot_filled_size / required_exit_size)
                if required_exit_size > 0.0
                else (1.0 if snapshot_status == "filled" and snapshot_filled_size > 0.0 else 0.0)
            )
            pending_exit["provider_status"] = snapshot_status or pending_exit.get("provider_status")
            if snapshot_filled_size > 0.0:
                pending_exit["filled_size"] = float(snapshot_filled_size)
            if snapshot_fill_price is not None and snapshot_fill_price > 0.0:
                pending_exit["average_fill_price"] = float(snapshot_fill_price)
            if fill_ratio > 0.0:
                pending_exit["fill_ratio"] = float(fill_ratio)
            if snapshot is not None:
                pending_exit["last_snapshot_at"] = _iso_utc(now)

            terminal_provider_status = snapshot_status in {"filled", "matched", "executed"}
            provider_status_unknown = snapshot_status in {"", "missing", "invalid", "unknown"}
            wallet_flat_confirmed = wallet_position_size <= _WALLET_SIZE_EPSILON and (
                wallet_position_observed
                or bool(wallet_positions_by_token)
                or isinstance(latest_wallet_sell_trade, dict)
            )
            close_fill_threshold_met = (
                required_exit_size > 0.0
                and terminal_provider_status
                and snapshot_filled_size >= (required_exit_size * threshold_ratio)
            )
            close_fill_threshold_with_wallet_confirmation = (
                required_exit_size > 0.0
                and provider_status_unknown
                and snapshot_filled_size >= (required_exit_size * threshold_ratio)
                and wallet_flat_confirmed
            )
            close_fill_terminal_with_wallet_confirmation = (
                required_exit_size > 0.0
                and terminal_provider_status
                and snapshot_filled_size > 0.0
                and wallet_flat_confirmed
            )
            close_fill_unknown_but_wallet_flat = (
                required_exit_size <= 0.0 and terminal_provider_status and wallet_flat_confirmed
            )
            if (
                close_fill_threshold_met
                or close_fill_threshold_with_wallet_confirmation
                or close_fill_terminal_with_wallet_confirmation
                or close_fill_unknown_but_wallet_flat
            ):
                base_qty = _fill_sz if _fill_sz > 0 else (_not / _ep if _ep and _ep > 0 else 0.0)
                close_qty = snapshot_filled_size if snapshot_filled_size > 0 else required_exit_size
                if close_qty <= 0.0:
                    close_qty = base_qty
                if base_qty <= 0.0 or _not <= 0.0:
                    held += 1
                    if not dry_run:
                        payload["pending_live_exit"] = pending_exit
                        _attach_pending_state(payload)
                        row.payload_json = payload
                        row.updated_at = now
                        state_updates += 1
                    continue
                close_qty = min(base_qty, close_qty if close_qty > 0.0 else base_qty)
                cost_basis = _not * (close_qty / base_qty) if base_qty > 0 else _not
                cp = snapshot_fill_price
                if cp is None or cp <= 0.0:
                    cp = safe_float(pending_exit.get("close_price")) or _ep
                close_trigger = str(pending_exit.get("close_trigger") or "live_exit_fill")
                _pnl = (close_qty * cp) - cost_basis
                ns = _status_for_close(pnl=_pnl, close_trigger=close_trigger)
                if not dry_run:
                    row.status = ns
                    row.actual_profit = _pnl
                    row.updated_at = now
                    pending_exit["status"] = "filled"
                    if close_fill_terminal_with_wallet_confirmation and not close_fill_threshold_met:
                        pending_exit["allow_partial_fill_terminal"] = True
                    pending_exit["filled_at"] = _iso_utc(now)
                    payload["pending_live_exit"] = pending_exit
                    payload["position_close"] = {
                        "close_price": cp,
                        "price_source": "provider_exit_fill",
                        "close_trigger": close_trigger,
                        "realized_pnl": _pnl,
                        "cost_basis_usd": cost_basis,
                        "settlement_proceeds_usd": close_qty * cp,
                        "filled_size": close_qty,
                        "closed_at": _iso_utc(now),
                        "reason": reason,
                    }
                    reverse_signal_id, reverse_source = await _emit_armed_reverse_signal(
                        session,
                        row=row,
                        payload=payload,
                        signal_payload=pending_signal_payload,
                        market_info=pending_market_info,
                        close_trigger=close_trigger,
                        realized_pnl=_pnl,
                        now=now,
                    )
                    if reverse_signal_id and reverse_source:
                        reverse_signal_ids_by_source.setdefault(reverse_source, []).append(reverse_signal_id)
                    row.payload_json = payload
                    closed += 1
                total_realized_pnl += _pnl
                by_status[ns] = int(by_status.get(ns, 0)) + 1
                would_close += 1
                details.append(
                    {
                        "order_id": row.id,
                        "market_id": row.market_id,
                        "close_trigger": close_trigger,
                        "close_price": cp,
                        "realized_pnl": _pnl,
                        "next_status": ns,
                    }
                )
                continue

            if snapshot_status in {"cancelled", "expired", "failed"}:
                if not dry_run:
                    pending_exit["status"] = "failed"
                    pending_exit["retry_count"] = int(pending_exit.get("retry_count", 0) or 0) + 1
                    pending_exit["last_attempt_at"] = _iso_utc(now)
                    pending_exit["last_error"] = f"provider_exit_status:{snapshot_status}"
                    pending_exit["next_retry_at"] = _iso_utc(
                        now + timedelta(seconds=_failed_exit_retry_delay_seconds(pending_exit["last_error"]))
                    )
                    payload["pending_live_exit"] = pending_exit
                    _attach_pending_state(payload)
                    row.payload_json = payload
                    row.updated_at = now
                    state_updates += 1
                held += 1
                continue

            if not dry_run and snapshot is not None:
                payload["pending_live_exit"] = pending_exit
                _attach_pending_state(payload)
                row.payload_json = payload
                row.updated_at = now
                state_updates += 1

            if pending_exit_kind != "take_profit_limit":
                if not dry_run and snapshot is None and pending_state_changed:
                    payload["position_state"] = pending_next_state
                    row.payload_json = payload
                    row.updated_at = now
                    state_updates += 1
                held += 1
                continue

        # ── Failed exit retry logic ────────────────────────────────────
        # If a previous cycle recorded a pending_live_exit that failed,
        # retry submitting the sell order (with cooldown & max retries).
        if isinstance(pending_exit, dict) and pending_exit.get("status") == "failed":
            if pending_winning_idx is not None and pending_outcome_idx is not None:
                _fill_not, _fill_sz, _fill_px = _extract_live_fill_metrics(payload)
                _ep = _fill_px if _fill_px and _fill_px > 0 else safe_float(row.effective_price)
                if _ep is None or _ep <= 0:
                    _ep = safe_float(row.entry_price)
                _not = _fill_not if _fill_not > 0 else (safe_float(row.notional_usd) or 0.0)
                _qty = _fill_sz if _fill_sz > 0 else (_not / _ep if _ep and _ep > 0 else 0.0)
                cp = 1.0 if pending_winning_idx == pending_outcome_idx else 0.0
                close_trigger = "resolution_inferred" if pending_winning_idx_inferred else "resolution"
                _pnl = (_qty * cp) - _not if _qty > 0 and _not > 0 else 0.0
                ns = _status_for_close(pnl=_pnl, close_trigger=close_trigger)
                if not dry_run:
                    row.status = ns
                    row.actual_profit = _pnl
                    row.updated_at = now
                    pending_exit["status"] = "superseded_resolution"
                    pending_exit["resolved_at"] = _iso_utc(now)
                    payload["pending_live_exit"] = pending_exit
                    payload["position_close"] = {
                        "close_price": cp,
                        "price_source": "resolved_settlement",
                        "close_trigger": close_trigger,
                        "realized_pnl": _pnl,
                        "closed_at": _iso_utc(now),
                        "reason": reason,
                    }
                    row.payload_json = payload
                    closed += 1
                total_realized_pnl += _pnl
                by_status[ns] = int(by_status.get(ns, 0)) + 1
                would_close += 1
                details.append(
                    {
                        "order_id": row.id,
                        "market_id": row.market_id,
                        "close_trigger": close_trigger,
                        "close_price": cp,
                        "realized_pnl": _pnl,
                        "next_status": ns,
                    }
                )
                continue

            if wallet_settlement_price is not None:
                _fill_not, _fill_sz, _fill_px = _extract_live_fill_metrics(payload)
                _ep = _fill_px if _fill_px and _fill_px > 0 else safe_float(row.effective_price)
                if _ep is None or _ep <= 0:
                    _ep = safe_float(row.entry_price)
                _not = _fill_not if _fill_not > 0 else (safe_float(row.notional_usd) or 0.0)
                _qty = _fill_sz if _fill_sz > 0 else (_not / _ep if _ep and _ep > 0 else 0.0)
                cp = wallet_settlement_price
                _pnl = (_qty * cp) - _not if _qty > 0 and _not > 0 else 0.0
                ns = _status_for_close(pnl=_pnl, close_trigger="resolution")
                if not dry_run:
                    row.status = ns
                    row.actual_profit = _pnl
                    row.updated_at = now
                    pending_exit["status"] = "superseded_resolution"
                    pending_exit["resolved_at"] = _iso_utc(now)
                    payload["pending_live_exit"] = pending_exit
                    payload["position_close"] = {
                        "close_price": cp,
                        "price_source": "wallet_redeemable_mark",
                        "close_trigger": "resolution",
                        "realized_pnl": _pnl,
                        "closed_at": _iso_utc(now),
                        "reason": reason,
                    }
                    row.payload_json = payload
                    closed += 1
                total_realized_pnl += _pnl
                by_status[ns] = int(by_status.get(ns, 0)) + 1
                would_close += 1
                details.append(
                    {
                        "order_id": row.id,
                        "market_id": row.market_id,
                        "close_trigger": "resolution",
                        "close_price": cp,
                        "realized_pnl": _pnl,
                        "next_status": ns,
                    }
                )
                continue

            retry_count = int(pending_exit.get("retry_count", 0) or 0)
            last_attempt_iso = pending_exit.get("last_attempt_at") or pending_exit.get("triggered_at")
            last_attempt_dt = _parse_iso_utc_naive(last_attempt_iso)
            next_retry_iso = pending_exit.get("next_retry_at")
            next_retry_dt = _parse_iso_utc_naive(next_retry_iso)
            min_retry_seconds = max(
                _FAILED_EXIT_MIN_RETRY_INTERVAL_SECONDS,
                _failed_exit_retry_delay_seconds(pending_exit.get("last_error")),
            )
            now_naive = now.astimezone(timezone.utc).replace(tzinfo=None)
            seconds_since_attempt = (
                (now_naive - last_attempt_dt).total_seconds() if last_attempt_dt is not None else float("inf")
            )

            if retry_count >= _FAILED_EXIT_MAX_RETRIES:
                if not dry_run:
                    pending_exit["status"] = "blocked_retry_exhausted"
                    pending_exit["exhausted_at"] = _iso_utc(now)
                    pending_exit["retry_count"] = retry_count
                    pending_exit["last_attempt_at"] = _iso_utc(now)
                    pending_exit["last_error"] = str(pending_exit.get("last_error") or "exit_retry_exhausted")
                    pending_exit["next_retry_at"] = None
                    payload["pending_live_exit"] = pending_exit
                    _attach_pending_state(payload)
                    row.payload_json = payload
                    row.updated_at = now
                    state_updates += 1
                held += 1
                continue

            if next_retry_dt is not None and now_naive < next_retry_dt:
                if not dry_run and pending_state_changed:
                    payload["position_state"] = pending_next_state
                    row.payload_json = payload
                    row.updated_at = now
                    state_updates += 1
                held += 1
                continue

            if seconds_since_attempt < min_retry_seconds:
                # Cooldown not elapsed — hold, don't spam.
                if not dry_run and pending_state_changed:
                    payload["position_state"] = pending_next_state
                    row.payload_json = payload
                    row.updated_at = now
                    state_updates += 1
                held += 1
                continue

            # Retry: refresh allowance and re-place exit order
            _fill_not_r, _fill_sz_r, _ = _extract_live_fill_metrics(payload)
            exit_size = max(
                max(0.0, safe_float(pending_exit.get("exit_size"), 0.0) or 0.0),
                _fill_sz_r if _fill_sz_r > 0 else 0.0,
            )
            wallet_exit_size_cap = wallet_position_size if wallet_position_size > _WALLET_SIZE_EPSILON else 0.0
            if exit_size <= 0.0 and wallet_exit_size_cap > 0.0:
                exit_size = wallet_exit_size_cap
            if exit_size > 0.0:
                pending_exit["exit_size"] = float(exit_size)
            exit_size = _remaining_exit_size(
                required_exit_size=exit_size,
                pending_exit=pending_exit,
                wallet_position_size=wallet_exit_size_cap,
            )
            if exit_size > 0.0:
                pending_exit["remaining_size"] = float(exit_size)
            exit_price = safe_float(pending_exit.get("close_price")) or 0.01
            min_order_size_usd = _resolve_position_min_order_size_usd(
                trader_params=params,
                payload=payload,
                mode="live",
            )

            if token_id and exit_size > 0:
                exit_notional_estimate = float(exit_size) * float(max(exit_price, 0.0))
                if exit_notional_estimate + 1e-9 < min_order_size_usd:
                    if not dry_run:
                        pending_exit["status"] = "blocked_min_notional"
                        pending_exit["retry_count"] = retry_count + 1
                        pending_exit["exhausted_at"] = _iso_utc(now)
                        pending_exit["last_attempt_at"] = _iso_utc(now)
                        pending_exit["last_error"] = (
                            f"exit_notional_below_min:{exit_notional_estimate:.4f}<{min_order_size_usd:.4f}"
                        )
                        pending_exit["next_retry_at"] = None
                        payload["pending_live_exit"] = pending_exit
                        _attach_pending_state(payload)
                        row.payload_json = payload
                        row.updated_at = now
                        state_updates += 1
                    held += 1
                    continue
                try:
                    from services.live_execution_adapter import execute_live_order

                    try:
                        await live_execution_service.prepare_sell_balance_allowance(token_id)
                    except Exception:
                        pass

                    exec_result = await execute_live_order(
                        token_id=token_id,
                        side="SELL",
                        size=exit_size,
                        fallback_price=exit_price,
                        min_order_size_usd=min_order_size_usd,
                        time_in_force="GTC",
                        resolve_live_price=False,
                    )
                    if exec_result.status in {"executed", "open", "submitted"}:
                        logger.info(
                            "Exit retry succeeded for order=%s attempt=%d status=%s",
                            row.id,
                            retry_count + 1,
                            exec_result.status,
                        )
                        if not dry_run:
                            pending_exit["status"] = "submitted"
                            pending_exit["retry_count"] = retry_count + 1
                            pending_exit["last_attempt_at"] = _iso_utc(now)
                            pending_exit["next_retry_at"] = None
                            pending_exit["exit_order_id"] = exec_result.order_id
                            pending_exit["provider_clob_order_id"] = str(
                                (exec_result.payload or {}).get("clob_order_id") or ""
                            )
                            payload["pending_live_exit"] = pending_exit
                            _attach_pending_state(payload)
                            row.payload_json = payload
                            row.updated_at = now
                            state_updates += 1
                        held += 1
                        continue
                    else:
                        logger.warning(
                            "Exit retry failed for order=%s attempt=%d error=%s",
                            row.id,
                            retry_count + 1,
                            exec_result.error_message,
                        )
                        if not dry_run:
                            pending_exit["status"] = "failed"
                            pending_exit["retry_count"] = retry_count + 1
                            pending_exit["last_attempt_at"] = _iso_utc(now)
                            pending_exit["last_error"] = str(exec_result.error_message or "unknown")
                            pending_exit["next_retry_at"] = _iso_utc(
                                now + timedelta(seconds=_failed_exit_retry_delay_seconds(pending_exit["last_error"]))
                            )
                            payload["pending_live_exit"] = pending_exit
                            _attach_pending_state(payload)
                            row.payload_json = payload
                            row.updated_at = now
                            state_updates += 1
                        held += 1
                        continue
                except Exception as exc:
                    logger.warning(
                        "Exit retry exception for order=%s attempt=%d: %s",
                        row.id,
                        retry_count + 1,
                        exc,
                    )
                    if not dry_run:
                        pending_exit["status"] = "failed"
                        pending_exit["retry_count"] = retry_count + 1
                        pending_exit["last_attempt_at"] = _iso_utc(now)
                        pending_exit["last_error"] = str(exc)
                        pending_exit["next_retry_at"] = _iso_utc(
                            now + timedelta(seconds=_failed_exit_retry_delay_seconds(pending_exit["last_error"]))
                        )
                        payload["pending_live_exit"] = pending_exit
                        _attach_pending_state(payload)
                        row.payload_json = payload
                        row.updated_at = now
                        state_updates += 1
                    held += 1
                    continue
            else:
                # No token_id or size — can't retry, mark exhausted
                if not dry_run:
                    pending_exit["status"] = "failed"
                    pending_exit["exhausted_at"] = _iso_utc(now)
                    pending_exit["retry_count"] = retry_count + 1
                    pending_exit["last_attempt_at"] = _iso_utc(now)
                    pending_exit["last_error"] = "missing token_id or fill_size"
                    pending_exit["next_retry_at"] = _iso_utc(
                        now + timedelta(seconds=_failed_exit_retry_delay_seconds(pending_exit["last_error"]))
                    )
                    payload["pending_live_exit"] = pending_exit
                    _attach_pending_state(payload)
                    row.payload_json = payload
                    row.updated_at = now
                    state_updates += 1
                held += 1
                continue

        if isinstance(pending_exit, dict) and pending_exit.get("status") in {
            "blocked_min_notional",
            "blocked_retry_exhausted",
        }:
            if pending_winning_idx is None and wallet_settlement_price is None:
                if not dry_run and pending_state_changed:
                    payload["position_state"] = pending_next_state
                    row.payload_json = payload
                    row.updated_at = now
                    state_updates += 1
                held += 1
                continue

        # If pending_live_exit is in-flight (submitted), skip normal processing
        if (
            isinstance(pending_exit, dict)
            and pending_exit.get("status") in {"submitted", "pending"}
            and str(pending_exit.get("kind") or "").strip().lower() != "take_profit_limit"
        ):
            if not dry_run and pending_state_changed:
                payload["position_state"] = pending_next_state
                row.payload_json = payload
                row.updated_at = now
                state_updates += 1
            held += 1
            continue

        filled_notional, filled_size, fill_price = _extract_live_fill_metrics(payload)
        status_key = str(row.status or "").strip().lower()
        if status_key in {"open", "submitted"} and filled_notional <= 0.0 and filled_size <= 0.0:
            skipped += 1
            skipped_reasons["awaiting_fill"] = int(skipped_reasons.get("awaiting_fill", 0)) + 1
            continue
        entry_price = fill_price if fill_price is not None and fill_price > 0 else safe_float(row.effective_price)
        if entry_price is None or entry_price <= 0:
            entry_price = safe_float(row.entry_price)
        notional = filled_notional if filled_notional > 0.0 else (safe_float(row.notional_usd) or 0.0)
        outcome_idx = _direction_outcome_index(row.direction)
        if outcome_idx is None or entry_price is None or entry_price <= 0 or notional <= 0:
            skipped += 1
            skipped_reasons["invalid_entry"] = int(skipped_reasons.get("invalid_entry", 0)) + 1
            continue

        signal_payload = signal_payloads.get(str(row.signal_id), {})
        market_info = market_info_by_id.get(str(row.market_id or ""))
        market_tradable = polymarket_client.is_market_tradable(market_info, now=now)
        market_seconds_left = _market_seconds_left(market_info, now)
        market_end_time = _market_end_time_iso(market_info)
        winning_idx = _extract_winning_outcome_index(market_info)
        winning_idx_inferred = False
        if winning_idx is None and resolution_infer_from_prices:
            inferred_idx = _extract_winning_outcome_index_from_prices(
                market_info,
                market_tradable=market_tradable,
                settle_floor=resolution_settle_floor,
            )
            if inferred_idx is not None:
                winning_idx = inferred_idx
                winning_idx_inferred = True
        market_side_price = _extract_market_side_price(market_info, outcome_idx)
        redis_side_price = redis_mid_prices.get(token_id) if token_id else None
        clob_side_price = clob_mid_prices.get(token_id) if token_id else None

        close_price: Optional[float] = None
        close_trigger: Optional[str] = None
        price_source: Optional[str] = None
        trailing_trigger_price: Optional[float] = None

        current_price = (
            redis_side_price
            if redis_side_price is not None
            else (
                clob_side_price
                if clob_side_price is not None
                else (
                    market_side_price
                    if market_side_price is not None
                    else wallet_mark_price
                )
            )
        )
        current_price = _state_price_floor(current_price)
        current_price_source = (
            "redis_mid"
            if redis_side_price is not None
            else (
                "clob_midpoint"
                if clob_side_price is not None
                else (
                    "market_mark"
                    if market_side_price is not None
                    else ("wallet_mark" if wallet_mark_price is not None else None)
                )
            )
        )

        age_anchor = row.executed_at or row.updated_at or row.created_at
        age_minutes = None
        if age_anchor is not None:
            age_minutes = max(0.0, (now - age_anchor).total_seconds() / 60.0)
        min_hold_passed = age_minutes is None or age_minutes >= min_hold_minutes

        position_state = _extract_position_state(payload)
        prev_high = safe_float(position_state.get("highest_price"))
        prev_low = safe_float(position_state.get("lowest_price"))
        prev_last_mark = safe_float(position_state.get("last_mark_price"))
        prev_mark_source = str(position_state.get("last_mark_source") or "")
        prev_marked_at = _parse_iso_utc_naive(position_state.get("last_marked_at"))
        highest_price = prev_high
        lowest_price = prev_low
        if current_price is not None:
            if highest_price is None:
                highest_price = current_price
            else:
                highest_price = max(highest_price, current_price)
            if lowest_price is None:
                lowest_price = current_price
            else:
                lowest_price = min(lowest_price, current_price)

        next_state = {
            "highest_price": _state_price_floor(highest_price),
            "lowest_price": _state_price_floor(lowest_price),
            "last_mark_price": current_price,
            "last_mark_source": current_price_source,
            "last_marked_at": _iso_utc(now),
        }
        prev_mark_age_seconds = (
            max(0.0, (now_naive - prev_marked_at).total_seconds()) if prev_marked_at is not None else None
        )
        fallback_mark_fresh = bool(
            prev_last_mark is not None
            and prev_mark_age_seconds is not None
            and prev_mark_age_seconds <= _MAX_LIVE_EXIT_FALLBACK_MARK_AGE_SECONDS
        )
        if current_price is not None:
            exit_eval_price = current_price
            exit_eval_price_source = current_price_source
        elif fallback_mark_fresh:
            exit_eval_price = _state_price_floor(prev_last_mark)
            exit_eval_price_source = "position_state_mark"
        else:
            exit_eval_price = None
            exit_eval_price_source = None

        if winning_idx is not None:
            close_price = 1.0 if winning_idx == outcome_idx else 0.0
            close_trigger = "resolution_inferred" if winning_idx_inferred else "resolution"
            price_source = "resolved_settlement"
        elif wallet_settlement_price is not None:
            close_price = wallet_settlement_price
            close_trigger = "resolution"
            price_source = "wallet_redeemable_mark"
        else:
            pnl_pct = None
            if exit_eval_price is not None and entry_price > 0:
                pnl_pct = ((exit_eval_price - entry_price) / entry_price) * 100.0

            if force_mark_to_market and exit_eval_price is not None:
                close_price = exit_eval_price
                close_trigger = "manual_mark_to_market"
                price_source = exit_eval_price_source
            else:
                # ── Strategy-based exit check ──────────────────────────
                # If the strategy that opened this position has a
                # should_exit() method, call it first and respect its
                # decision before falling through to default TP/SL/etc.
                strategy_slug = (payload.get("strategy_type") or "").strip().lower()
                strategy_exit = None
                if strategy_slug:
                    from services.strategy_loader import strategy_loader

                    loaded = strategy_loader.get_strategy(strategy_slug)
                    if loaded and hasattr(loaded.instance, "should_exit"):
                        try:

                            class _LivePositionView:
                                pass

                            pos_view = _LivePositionView()
                            pos_view.entry_price = entry_price
                            pos_view.current_price = exit_eval_price
                            pos_view.highest_price = highest_price
                            pos_view.lowest_price = lowest_price
                            pos_view.age_minutes = age_minutes
                            pos_view.pnl_percent = pnl_pct
                            pos_view.filled_size = (
                                filled_size
                                if filled_size > 0.0
                                else (notional / entry_price if entry_price > 0 else 0.0)
                            )
                            pos_view.notional_usd = notional
                            if "strategy_context" not in payload:
                                payload["strategy_context"] = {}
                            pos_view.strategy_context = payload["strategy_context"]
                            pos_view.config = payload.get("strategy_exit_config", {})
                            pos_view.outcome_idx = outcome_idx

                            min_order_size_usd = _resolve_position_min_order_size_usd(
                                trader_params=params,
                                payload=payload,
                                mode="live",
                            )
                            market_state_dict = {
                                "current_price": exit_eval_price,
                                "market_tradable": market_tradable,
                                "is_resolved": False,
                                "winning_outcome": None,
                                "seconds_left": market_seconds_left,
                                "end_time": market_end_time,
                                "token_id": token_id,
                                "mark_source": exit_eval_price_source,
                                "min_order_size_usd": min_order_size_usd,
                                "notional_usd": notional,
                            }

                            exit_decision = loaded.instance.should_exit(pos_view, market_state_dict)
                            exit_action = getattr(exit_decision, "action", None) if exit_decision is not None else None
                            if exit_action == "close":
                                strategy_exit = exit_decision
                            elif exit_action == "reduce":
                                strategy_exit = exit_decision
                        except Exception as exc:
                            logger.warning(
                                "Strategy should_exit() error for %s: %s",
                                strategy_slug,
                                exc,
                            )

                if strategy_exit is not None and getattr(strategy_exit, "action", None) == "reduce":
                    if not dry_run:
                        payload["position_state"] = next_state
                        row.payload_json = payload
                        row.updated_at = now
                        state_updates += 1
                    held += 1
                    continue

                active_take_profit_limit = (
                    isinstance(pending_exit, dict)
                    and str(pending_exit.get("kind") or "").strip().lower() == "take_profit_limit"
                    and str(pending_exit.get("status") or "").strip().lower() in {"submitted", "pending"}
                )
                if strategy_exit is not None:
                    _arm_reverse_entry_from_exit(
                        row=row,
                        payload=payload,
                        strategy_exit=strategy_exit,
                        market_info=market_info,
                        market_seconds_left=market_seconds_left,
                        current_price=exit_eval_price,
                        now=now,
                    )
                    close_price = (
                        strategy_exit.close_price if strategy_exit.close_price is not None else exit_eval_price
                    )
                    close_trigger = f"strategy:{strategy_exit.reason}"
                    price_source = exit_eval_price_source
                elif (
                    not resolve_only
                    and take_profit_pct is not None
                    and pnl_pct is not None
                    and min_hold_passed
                    and not active_take_profit_limit
                    and pnl_pct >= take_profit_pct
                ):
                    close_price = exit_eval_price
                    close_trigger = "take_profit"
                    price_source = exit_eval_price_source
                elif (
                    not resolve_only
                    and stop_loss_pct is not None
                    and pnl_pct is not None
                    and min_hold_passed
                    and pnl_pct <= -abs(stop_loss_pct)
                ):
                    close_price = exit_eval_price
                    close_trigger = "stop_loss"
                    price_source = exit_eval_price_source
                elif (
                    not resolve_only
                    and trailing_stop_pct is not None
                    and trailing_stop_pct > 0
                    and exit_eval_price is not None
                    and highest_price is not None
                    and min_hold_passed
                ):
                    trailing_trigger_price = highest_price * (1.0 - (trailing_stop_pct / 100.0))
                    if highest_price > entry_price and exit_eval_price <= trailing_trigger_price:
                        close_price = exit_eval_price
                        close_trigger = "trailing_stop"
                        price_source = exit_eval_price_source
                elif (
                    not resolve_only
                    and max_hold_minutes is not None
                    and age_minutes is not None
                    and age_minutes >= max_hold_minutes
                ):
                    if exit_eval_price is not None:
                        close_price = exit_eval_price
                        close_trigger = "max_hold"
                        price_source = exit_eval_price_source
                elif (
                    not resolve_only
                    and close_on_inactive_market
                    and not market_tradable
                    and exit_eval_price is not None
                    and min_hold_passed
                ):
                    close_price = exit_eval_price
                    close_trigger = "market_inactive"
                    price_source = exit_eval_price_source

        close_is_resolution = close_trigger in {"resolution", "resolution_inferred"}

        if close_price is None:
            state_changed = False
            if current_price is not None:
                mark_stale = (
                    prev_marked_at is None
                    or (now_naive - prev_marked_at).total_seconds() >= mark_touch_interval_seconds
                )
                state_changed = (
                    prev_last_mark is None
                    or abs(prev_last_mark - current_price) > 1e-9
                    or prev_high is None
                    or prev_low is None
                    or abs((prev_high or 0.0) - (highest_price or 0.0)) > 1e-9
                    or abs((prev_low or 0.0) - (lowest_price or 0.0)) > 1e-9
                    or prev_mark_source != str(current_price_source or "")
                    or mark_stale
                )
            if not dry_run and state_changed:
                payload["position_state"] = next_state
                row.payload_json = payload
                row.updated_at = now
                state_updates += 1
            held += 1
            continue

        quantity = filled_size if filled_size > 0.0 else (notional / entry_price if entry_price > 0 else 0.0)
        cost_basis = filled_notional if filled_notional > 0.0 else notional
        if quantity <= 0.0 or cost_basis <= 0.0:
            skipped += 1
            skipped_reasons["invalid_fill_state"] = int(skipped_reasons.get("invalid_fill_state", 0)) + 1
            continue
        proceeds = quantity * close_price
        pnl = proceeds - cost_basis
        next_status = _status_for_close(pnl=pnl, close_trigger=close_trigger)

        detail = {
            "order_id": row.id,
            "market_id": row.market_id,
            "direction": row.direction,
            "entry_price": entry_price,
            "close_price": close_price,
            "price_source": price_source,
            "close_trigger": close_trigger,
            "market_tradable": market_tradable,
            "notional_usd": notional,
            "cost_basis_usd": cost_basis,
            "filled_size": filled_size,
            "filled_notional_usd": filled_notional,
            "quantity": quantity,
            "next_status": next_status,
            "age_minutes": age_minutes,
            "min_hold_minutes": min_hold_minutes,
            "trailing_stop_trigger_price": trailing_trigger_price,
            "highest_price_seen": _state_price_floor(highest_price),
            "lowest_price_seen": _state_price_floor(lowest_price),
        }

        if not close_is_resolution:
            detail["next_status"] = str(row.status or "").strip().lower()
            detail["realized_pnl"] = None
            detail["hypothetical_pnl"] = pnl
            details.append(detail)
            would_close += 1

            if not dry_run:
                payload["position_state"] = next_state
                existing_tp_limit = (
                    pending_exit
                    if isinstance(pending_exit, dict)
                    and str(pending_exit.get("kind") or "").strip().lower() == "take_profit_limit"
                    and str(pending_exit.get("status") or "").strip().lower() in {"submitted", "pending"}
                    else None
                )
                if existing_tp_limit is not None:
                    cancel_target = _pending_exit_provider_clob_id(existing_tp_limit)
                    if not cancel_target:
                        cancel_target = str(existing_tp_limit.get("exit_order_id") or "").strip()
                    cancel_success = False
                    if cancel_target:
                        try:
                            cancel_success = bool(await live_execution_service.cancel_order(cancel_target))
                        except Exception:
                            cancel_success = False
                    existing_tp_limit["cancelled_for_override_at"] = _iso_utc(now)
                    existing_tp_limit["override_cancel_target"] = cancel_target
                    existing_tp_limit["override_cancel_success"] = cancel_success
                    if cancel_success:
                        existing_tp_limit["status"] = "cancelled"
                    payload["superseded_take_profit_exit"] = existing_tp_limit

                exit_record: dict[str, Any] = {
                    "triggered_at": _iso_utc(now),
                    "close_trigger": close_trigger,
                    "close_price": close_price,
                    "price_source": price_source,
                    "market_tradable": market_tradable,
                    "hypothetical_pnl": pnl,
                    "age_minutes": age_minutes,
                    "reason": reason,
                    "retry_count": 0,
                    "status": "pending",
                }

                # Immediately attempt to place the sell order
                token_id = _extract_live_token_id(payload)
                exit_size = filled_size if filled_size > 0.0 else quantity
                if exit_size > 0.0:
                    exit_record["exit_size"] = float(exit_size)
                wallet_exit_size_cap = wallet_position_size if wallet_position_size > _WALLET_SIZE_EPSILON else 0.0
                if wallet_exit_size_cap > 0.0:
                    if exit_size <= 0.0:
                        exit_size = wallet_exit_size_cap
                    else:
                        exit_size = min(exit_size, wallet_exit_size_cap)
                    exit_record["exit_size"] = float(exit_size)
                min_order_size_usd = _resolve_position_min_order_size_usd(
                    trader_params=params,
                    payload=payload,
                    mode="live",
                )
                if token_id and exit_size > 0:
                    exit_size = _remaining_exit_size(
                        required_exit_size=exit_size,
                        pending_exit=exit_record,
                        wallet_position_size=wallet_exit_size_cap,
                    )
                    if exit_size > 0.0:
                        exit_record["remaining_size"] = float(exit_size)
                    exit_notional_estimate = float(exit_size) * float(max(close_price, 0.0))
                    if exit_notional_estimate + 1e-9 < min_order_size_usd:
                        exit_record["status"] = "blocked_min_notional"
                        exit_record["retry_count"] = 1
                        exit_record["exhausted_at"] = _iso_utc(now)
                        exit_record["last_error"] = (
                            f"exit_notional_below_min:{exit_notional_estimate:.4f}<{min_order_size_usd:.4f}"
                        )
                        exit_record["last_attempt_at"] = _iso_utc(now)
                        exit_record["next_retry_at"] = None
                        payload["pending_live_exit"] = exit_record
                        row.payload_json = payload
                        row.updated_at = now
                        state_updates += 1
                        held += 1
                        continue
                    try:
                        from services.live_execution_adapter import execute_live_order

                        try:
                            await live_execution_service.prepare_sell_balance_allowance(token_id)
                        except Exception:
                            pass

                        exec_result = await execute_live_order(
                            token_id=token_id,
                            side="SELL",
                            size=exit_size,
                            fallback_price=close_price,
                            min_order_size_usd=min_order_size_usd,
                            time_in_force="GTC",
                            resolve_live_price=False,
                        )
                        if exec_result.status in {"executed", "open", "submitted"}:
                            exit_record["status"] = "submitted"
                            exit_record["exit_order_id"] = exec_result.order_id
                            exit_record["provider_clob_order_id"] = str(
                                (exec_result.payload or {}).get("clob_order_id") or ""
                            )
                            exit_record["last_attempt_at"] = _iso_utc(now)
                            logger.info(
                                "Exit order placed for order=%s trigger=%s status=%s",
                                row.id,
                                close_trigger,
                                exec_result.status,
                            )
                        else:
                            exit_record["status"] = "failed"
                            exit_record["last_error"] = str(exec_result.error_message or "unknown")
                            exit_record["last_attempt_at"] = _iso_utc(now)
                            exit_record["retry_count"] = 1
                            exit_record["next_retry_at"] = _iso_utc(
                                now + timedelta(seconds=_failed_exit_retry_delay_seconds(exit_record["last_error"]))
                            )
                            logger.warning(
                                "Exit order failed for order=%s trigger=%s error=%s",
                                row.id,
                                close_trigger,
                                exec_result.error_message,
                            )
                    except Exception as exc:
                        exit_record["status"] = "failed"
                        exit_record["last_error"] = str(exc)
                        exit_record["last_attempt_at"] = _iso_utc(now)
                        exit_record["retry_count"] = 1
                        exit_record["next_retry_at"] = _iso_utc(
                            now + timedelta(seconds=_failed_exit_retry_delay_seconds(exit_record["last_error"]))
                        )
                        logger.warning(
                            "Exit order exception for order=%s trigger=%s: %s",
                            row.id,
                            close_trigger,
                            exc,
                        )
                else:
                    exit_record["status"] = "failed"
                    exit_record["last_error"] = "missing token_id or fill_size"
                    exit_record["retry_count"] = 1
                    exit_record["next_retry_at"] = _iso_utc(
                        now + timedelta(seconds=_failed_exit_retry_delay_seconds(exit_record["last_error"]))
                    )

                payload["pending_live_exit"] = exit_record
                row.payload_json = payload
                row.updated_at = now
                state_updates += 1
            held += 1
            continue

        # NOTE: No simulation ledger interaction for live positions.
        # Live positions settle against real exchange state.
        total_realized_pnl += pnl
        by_status[next_status] = int(by_status.get(next_status, 0)) + 1
        detail["realized_pnl"] = pnl
        details.append(detail)
        would_close += 1

        if dry_run:
            continue

        row.status = next_status
        row.actual_profit = pnl
        row.updated_at = now
        payload["position_state"] = next_state
        payload["position_close"] = {
            "close_price": close_price,
            "price_source": price_source,
            "close_trigger": close_trigger,
            "realized_pnl": pnl,
            "cost_basis_usd": cost_basis,
            "settlement_proceeds_usd": proceeds,
            "filled_size": filled_size,
            "filled_notional_usd": filled_notional,
            "market_tradable": market_tradable,
            "age_minutes": age_minutes,
            "closed_at": _iso_utc(now),
            "reason": reason,
        }
        reverse_signal_id, reverse_source = await _emit_armed_reverse_signal(
            session,
            row=row,
            payload=payload,
            signal_payload=signal_payload,
            market_info=market_info,
            close_trigger=close_trigger,
            realized_pnl=pnl,
            now=now,
        )
        if reverse_signal_id and reverse_source:
            reverse_signal_ids_by_source.setdefault(reverse_source, []).append(reverse_signal_id)
        row.payload_json = payload
        if reason:
            if row.reason:
                row.reason = f"{row.reason} | {reason}:{close_trigger}"
            else:
                row.reason = f"{reason}:{close_trigger}"
        closed += 1

    if not dry_run and (closed > 0 or state_updates > 0):
        await session.commit()
        await _publish_trader_order_updates([row for row in candidates if row.updated_at == now])
        if reverse_signal_ids_by_source:
            await refresh_trade_signal_snapshots(session)
            await _publish_reverse_signal_batches(reverse_signal_ids_by_source, emitted_at=now)

    return {
        "trader_id": trader_id,
        "mode": "live",
        "dry_run": bool(dry_run),
        "matched": len(candidates),
        "would_close": would_close,
        "closed": closed,
        "held": held,
        "skipped": skipped,
        "state_updates": state_updates,
        "total_realized_pnl": total_realized_pnl,
        "by_status": by_status,
        "skipped_reasons": skipped_reasons,
        "reverse_signals_emitted": sum(len(ids) for ids in reverse_signal_ids_by_source.values()),
        "details": details,
    }
