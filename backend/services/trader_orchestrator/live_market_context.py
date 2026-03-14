from __future__ import annotations

import asyncio
import re
import time
from datetime import datetime, timezone
from typing import Any, Optional

from config import settings
from services.market_runtime import get_market_runtime
from services.polymarket import polymarket_client
from services.ws_feeds import get_feed_manager
from utils.converters import coerce_bool as _coerce_bool, normalize_identifier as _normalize_identifier, safe_float
from utils.logger import get_logger
from utils.signal_helpers import signal_payload
from utils.utcnow import utcnow

logger = get_logger(__name__)

_POLYMARKET_CONDITION_ID_RE = re.compile(r"^0x[0-9a-f]{64}$")
_POLYMARKET_NUMERIC_TOKEN_ID_RE = re.compile(r"^\d{18,}$")
_POLYMARKET_HEX_TOKEN_ID_RE = re.compile(r"^(?:0x)?[0-9a-f]{40,}$")


def _strict_ws_ttl_seconds_for_source(source: Any) -> float:
    default_ttl = max(0.05, float(getattr(settings, "WS_EXECUTION_PRICE_STALE_SECONDS", 1.0) or 1.0))
    normalized_source = str(source or "").strip().lower()
    if normalized_source != "scanner":
        return default_ttl
    try:
        scanner_max_age_ms = float(getattr(settings, "SCANNER_STRICT_WS_MAX_AGE_MS", 30000) or 30000)
    except Exception:
        scanner_max_age_ms = 30000.0
    return max(default_ttl, max(100.0, scanner_max_age_ms) / 1000.0)


def _parse_market_datetime(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, (int, float)):
        ts = float(value)
        if ts > 1_000_000_000_000:
            ts /= 1000.0
        if ts <= 0:
            return None
        try:
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        except Exception:
            return None
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        numeric = safe_float(raw)
        if numeric is not None:
            return _parse_market_datetime(numeric)
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except Exception:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    return None


def _extract_market_timing(market_info: dict[str, Any]) -> dict[str, Any]:
    end_dt: Optional[datetime] = None
    for key in (
        "end_time",
        "endTime",
        "end_date_iso",
        "endDateIso",
        "end_date",
        "endDate",
        "closes_at",
        "closesAt",
        "accepting_orders_until",
        "acceptingOrdersUntil",
    ):
        end_dt = _parse_market_datetime(market_info.get(key))
        if end_dt is not None:
            break

    seconds_left = safe_float(
        market_info.get("seconds_left")
        or market_info.get("secondsLeft")
        or market_info.get("seconds_until_close")
        or market_info.get("secondsUntilClose")
    )
    if seconds_left is not None:
        seconds_left = max(0.0, float(seconds_left))
    if end_dt is not None:
        computed = max(0.0, (end_dt - utcnow()).total_seconds())
        seconds_left = computed if seconds_left is None else min(seconds_left, computed)

    closed_hint = _coerce_bool(market_info.get("closed"))
    resolved_hint = _coerce_bool(market_info.get("resolved"))
    active_hint = _coerce_bool(market_info.get("active"))
    is_live = _coerce_bool(market_info.get("is_live"))
    if is_live is None:
        if closed_hint is True or resolved_hint is True:
            is_live = False
        elif seconds_left is not None:
            is_live = seconds_left > 0.0
        elif active_hint is not None:
            is_live = active_hint
    is_current = _coerce_bool(market_info.get("is_current"))
    if is_current is None:
        is_current = is_live

    return {
        "end_time": end_dt.isoformat().replace("+00:00", "Z") if end_dt is not None else None,
        "seconds_left": seconds_left,
        "is_live": is_live,
        "is_current": is_current,
        "closed": closed_hint,
        "resolved": resolved_hint,
    }


def _normalize_identifier(value: Any) -> str:
    return str(value or "").strip().lower()


def _iso_from_epoch_seconds(epoch_seconds: float | None) -> Optional[str]:
    if epoch_seconds is None:
        return None
    try:
        return datetime.fromtimestamp(float(epoch_seconds), tz=timezone.utc).isoformat().replace("+00:00", "Z")
    except Exception:
        return None


def _epoch_seconds_from_market_timestamp(value: Any) -> Optional[float]:
    parsed = _parse_market_datetime(value)
    if parsed is None:
        return None
    return parsed.timestamp()


def _is_condition_id(value: str) -> bool:
    return bool(_POLYMARKET_CONDITION_ID_RE.fullmatch(value))


def _is_token_id(value: str) -> bool:
    if _is_condition_id(value):
        return False
    return bool(_POLYMARKET_NUMERIC_TOKEN_ID_RE.fullmatch(value) or _POLYMARKET_HEX_TOKEN_ID_RE.fullmatch(value))


def _extract_token_ids(market_info: dict[str, Any]) -> list[str]:
    raw = market_info.get("token_ids") or market_info.get("clob_token_ids") or market_info.get("tokenIds") or []
    out: list[str] = []
    if not isinstance(raw, list):
        return out
    for token_id in raw:
        norm = _normalize_identifier(token_id)
        if not norm:
            continue
        out.append(norm)
    return out


def _extract_yes_no_tokens(market_info: dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
    tokens = _extract_token_ids(market_info)
    if not tokens:
        return None, None

    outcomes = market_info.get("outcomes")
    if not isinstance(outcomes, list):
        outcomes = market_info.get("outcome_labels") or market_info.get("outcomeLabels")
    if not isinstance(outcomes, list):
        outcomes = []

    yes_index: Optional[int] = None
    no_index: Optional[int] = None
    for idx, outcome in enumerate(outcomes):
        label = str(outcome or "").strip().lower()
        if label == "yes" and yes_index is None:
            yes_index = idx
        elif label == "no" and no_index is None:
            no_index = idx

    yes_token: Optional[str] = None
    no_token: Optional[str] = None
    if yes_index is not None and 0 <= yes_index < len(tokens):
        yes_token = tokens[yes_index]
    if no_index is not None and 0 <= no_index < len(tokens):
        no_token = tokens[no_index]

    if yes_token is None and tokens:
        yes_token = tokens[0]
    if no_token is None and len(tokens) > 1:
        no_token = tokens[1]
    return yes_token, no_token


def _extract_payload_market_hint(signal: Any) -> dict[str, Any]:
    payload_json = getattr(signal, "payload_json", None)
    payload = dict(payload_json) if isinstance(payload_json, dict) else {}
    merged_payload = signal_payload(signal)
    signal_market_id = _normalize_identifier(getattr(signal, "market_id", None))
    source_item_id = _normalize_identifier(getattr(signal, "source_item_id", None))
    lookup_ids = {value for value in (signal_market_id, source_item_id) if value}

    hint: dict[str, Any] = {}
    live_market = payload.get("live_market")
    if isinstance(live_market, dict):
        hint.update(dict(live_market))

    markets = payload.get("markets") if isinstance(payload.get("markets"), list) else []
    matched_market: dict[str, Any] | None = None
    for raw_market in markets:
        if not isinstance(raw_market, dict):
            continue
        candidate = dict(raw_market)
        candidate_ids = {
            _normalize_identifier(candidate.get("condition_id")),
            _normalize_identifier(candidate.get("conditionId")),
            _normalize_identifier(candidate.get("id")),
            _normalize_identifier(candidate.get("market_id")),
        }
        candidate_tokens = {
            _normalize_identifier(token_id)
            for token_id in (
                candidate.get("token_ids")
                or candidate.get("clob_token_ids")
                or candidate.get("tokenIds")
                or []
            )
            if _normalize_identifier(token_id)
        }
        if lookup_ids.intersection(candidate_ids) or lookup_ids.intersection(candidate_tokens):
            matched_market = candidate
            break
        if matched_market is None:
            matched_market = candidate
    if matched_market is not None:
        hint.update(matched_market)

    synthetic_hint: dict[str, Any] = {}
    for key in (
        "condition_id",
        "conditionId",
        "market_id",
        "id",
        "question",
        "market_question",
        "yes_price",
        "no_price",
        "up_price",
        "down_price",
        "liquidity",
        "volume",
        "spread",
        "oracle_price",
        "oracle_source",
        "price_to_beat",
        "source_observed_at",
        "price_updated_at",
        "updated_at",
        "fetched_at",
        "live_market_fetched_at",
        "history_tail",
        "seconds_left",
        "is_live",
        "is_current",
        "closed",
        "resolved",
        "active",
    ):
        value = merged_payload.get(key)
        if value not in (None, ""):
            synthetic_hint[key] = value

    selected_token = _normalize_identifier(
        merged_payload.get("selected_token_id")
        or merged_payload.get("token_id")
        or merged_payload.get("clob_token_id")
    )
    yes_token = _normalize_identifier(merged_payload.get("yes_token_id"))
    no_token = _normalize_identifier(merged_payload.get("no_token_id"))
    token_ids = [token_id for token_id in (yes_token, no_token) if token_id]
    if not token_ids and selected_token:
        token_ids = [selected_token]
    if token_ids:
        synthetic_hint.setdefault("token_ids", token_ids)

    selected_outcome = str(
        merged_payload.get("selected_outcome")
        or merged_payload.get("outcome")
        or merged_payload.get("side")
        or ""
    ).strip()
    if selected_outcome:
        synthetic_hint.setdefault("outcomes", [selected_outcome])

    if signal_market_id:
        synthetic_hint.setdefault("market_id", signal_market_id)
        synthetic_hint.setdefault("id", signal_market_id)
    if source_item_id and _is_condition_id(source_item_id):
        synthetic_hint.setdefault("condition_id", source_item_id)
    market_question = str(
        merged_payload.get("market_question")
        or getattr(signal, "market_question", "")
        or ""
    ).strip()
    if market_question:
        synthetic_hint.setdefault("question", market_question)

    if not synthetic_hint.get("token_ids"):
        positions = payload.get("positions_to_take") if isinstance(payload.get("positions_to_take"), list) else []
        for raw_position in positions:
            if not isinstance(raw_position, dict):
                continue
            token_id = _normalize_identifier(raw_position.get("token_id") or raw_position.get("clob_token_id"))
            if not token_id:
                continue
            outcome = str(raw_position.get("outcome") or raw_position.get("label") or "").strip()
            market_id = _normalize_identifier(raw_position.get("market_id") or raw_position.get("id"))
            if market_id:
                synthetic_hint.setdefault("market_id", market_id)
                synthetic_hint.setdefault("id", market_id)
            synthetic_hint["token_ids"] = [token_id]
            if outcome:
                synthetic_hint.setdefault("outcomes", [outcome])
            price = safe_float(raw_position.get("price"), None)
            if price is not None:
                if outcome.strip().lower() == "yes":
                    synthetic_hint.setdefault("yes_price", float(price))
                elif outcome.strip().lower() == "no":
                    synthetic_hint.setdefault("no_price", float(price))
            break

    if not synthetic_hint.get("token_ids"):
        execution_plan = payload.get("execution_plan")
        if isinstance(execution_plan, dict):
            raw_legs = execution_plan.get("legs") if isinstance(execution_plan.get("legs"), list) else []
            selected_leg = next(
                (
                    leg
                    for leg in raw_legs
                    if isinstance(leg, dict) and str(leg.get("side") or "").strip().lower() == "buy"
                ),
                None,
            )
            if selected_leg is None:
                selected_leg = next((leg for leg in raw_legs if isinstance(leg, dict)), None)
            if isinstance(selected_leg, dict):
                token_id = _normalize_identifier(selected_leg.get("token_id"))
                outcome = str(selected_leg.get("outcome") or "").strip()
                market_id = _normalize_identifier(selected_leg.get("market_id"))
                if market_id:
                    synthetic_hint.setdefault("market_id", market_id)
                    synthetic_hint.setdefault("id", market_id)
                if token_id:
                    synthetic_hint["token_ids"] = [token_id]
                if outcome:
                    synthetic_hint.setdefault("outcomes", [outcome])

    hint.update(synthetic_hint)
    return hint


def _signal_market_lookup_candidates(signal: Any, payload_hint: dict[str, Any]) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    token_ids = payload_hint.get("token_ids") or payload_hint.get("clob_token_ids") or payload_hint.get("tokenIds") or []
    for raw_value in (
        getattr(signal, "market_id", None),
        getattr(signal, "source_item_id", None),
        payload_hint.get("condition_id"),
        payload_hint.get("conditionId"),
        payload_hint.get("market_id"),
        payload_hint.get("id"),
        *token_ids,
    ):
        normalized = _normalize_identifier(raw_value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        candidates.append(normalized)
    if not candidates:
        candidates.append("")
    return candidates


def _extract_signal_market_hint(signal: Any) -> dict[str, Any]:
    payload_hint = _extract_payload_market_hint(signal)
    try:
        runtime = get_market_runtime()
    except Exception:
        return payload_hint
    for lookup_value in _signal_market_lookup_candidates(signal, payload_hint):
        snapshot = runtime.get_market_snapshot(lookup_value, hint=payload_hint)
        if isinstance(snapshot, dict) and snapshot:
            hinted_snapshot = dict(payload_hint)
            hinted_snapshot.update(dict(snapshot))
            hinted_snapshot["_runtime_market_snapshot"] = True
            return hinted_snapshot
    return payload_hint


def _normalize_history_points(
    points: list[dict[str, Any]],
    *,
    max_points: int,
) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    for item in points:
        if not isinstance(item, dict):
            continue
        t = safe_float(item.get("t"))
        p = safe_float(item.get("p"))
        if t is None or p is None:
            continue
        if t < 10_000_000_000:
            t *= 1000.0
        if not (0.0 <= p <= 1.01):
            continue
        out.append({"t": float(t), "p": float(p)})
    out.sort(key=lambda row: row["t"])
    if max_points > 0 and len(out) > max_points:
        out = out[-max_points:]
    return out


def _window_change(
    history: list[dict[str, float]],
    *,
    now_ms: float,
    window_seconds: int,
) -> dict[str, Optional[float]]:
    if len(history) < 2:
        return {"delta": None, "percent": None}

    cutoff = now_ms - (max(1, int(window_seconds)) * 1000.0)
    baseline = history[0]["p"]
    for point in history:
        if point["t"] >= cutoff:
            baseline = point["p"]
            break

    latest = history[-1]["p"]
    delta = latest - baseline
    pct = None
    if baseline > 0:
        pct = (delta / baseline) * 100.0
    return {"delta": delta, "percent": pct}


def _build_history_summary(history: list[dict[str, float]]) -> dict[str, Any]:
    now_ms = float(time.time() * 1000.0)
    if not history:
        return {
            "points": 0,
            "start_ts": None,
            "end_ts": None,
            "move_5m": {"delta": None, "percent": None},
            "move_30m": {"delta": None, "percent": None},
            "move_2h": {"delta": None, "percent": None},
        }
    return {
        "points": len(history),
        "start_ts": history[0]["t"],
        "end_ts": history[-1]["t"],
        "move_5m": _window_change(history, now_ms=now_ms, window_seconds=300),
        "move_30m": _window_change(history, now_ms=now_ms, window_seconds=1800),
        "move_2h": _window_change(history, now_ms=now_ms, window_seconds=7200),
    }


def _extract_model_probability(signal: Any, *, direction: str) -> Optional[float]:
    # Generic fallback for directional signals where edge = model_probability - entry_price.
    entry = safe_float(getattr(signal, "entry_price", None))
    edge = safe_float(getattr(signal, "edge_percent", None))
    if entry is not None and edge is not None:
        implied = entry + (edge / 100.0)
        if 0.0 <= implied <= 1.0:
            return implied

    payload = getattr(signal, "payload_json", None)
    if not isinstance(payload, dict):
        return None

    for key in ("model_probability", "selected_probability"):
        value = safe_float(payload.get(key))
        if value is not None and 0.0 <= value <= 1.0:
            return value

    # Common field for directional intents.
    value = safe_float(payload.get("expected_payout"))
    if value is not None and 0.0 <= value <= 1.0:
        return value

    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return None

    weather = metadata.get("weather")
    if isinstance(weather, dict):
        consensus_yes = safe_float(weather.get("consensus_probability"))
        if consensus_yes is not None and 0.0 <= consensus_yes <= 1.0:
            return (1.0 - consensus_yes) if direction == "buy_no" else consensus_yes

    for key in ("model_probability", "selected_probability"):
        value = safe_float(metadata.get(key))
        if value is not None and 0.0 <= value <= 1.0:
            return value

    return None


def _cached_history_points_for_token(
    feed_manager: Any,
    token_id: str,
    *,
    max_history_points: int,
) -> list[dict[str, float]]:
    if feed_manager is None or not getattr(feed_manager, "_started", False):
        return []
    try:
        cached_history = feed_manager.cache.get_price_history(
            token_id,
            max_snapshots=max(20, int(max_history_points) * 2),
        )
    except Exception as exc:
        logger.warning(
            "WebSocket history lookup failed",
            token_id=token_id,
            exc_info=exc,
        )
        return []
    normalized_cached: list[dict[str, Any]] = []
    for item in cached_history:
        if not isinstance(item, dict):
            continue
        ts = _epoch_seconds_from_market_timestamp(item.get("timestamp"))
        mid = safe_float(item.get("mid"))
        if ts is None or mid is None:
            continue
        normalized_cached.append({"t": ts * 1000.0, "p": float(mid)})
    return _normalize_history_points(
        normalized_cached,
        max_points=max_history_points,
    )


def _build_context_from_cached_market_state(
    signal: Any,
    *,
    market_id: str,
    market_info: dict[str, Any],
    direction: str,
    yes_token: str | None,
    no_token: str | None,
    yes_live: float | None,
    no_live: float | None,
    yes_age_ms: float | None,
    no_age_ms: float | None,
    yes_observed_at: str | None,
    no_observed_at: str | None,
    selected_token: str | None,
    selected_history: list[dict[str, float]],
) -> dict[str, Any]:
    signal_market_id = _normalize_identifier(getattr(signal, "market_id", ""))
    signal_entry = safe_float(getattr(signal, "entry_price", None))
    selected_outcome: Optional[str] = None
    selected_live: Optional[float] = None
    selected_source: str | None = None
    selected_age_ms: float | None = None
    selected_observed_at: str | None = None
    yes_source = "ws_strict" if yes_live is not None else None
    no_source = "ws_strict" if no_live is not None else None

    if direction == "buy_yes":
        selected_outcome = "yes"
        selected_live = yes_live
        selected_source = yes_source
        selected_age_ms = yes_age_ms
        selected_observed_at = yes_observed_at
    elif direction == "buy_no":
        selected_outcome = "no"
        selected_live = no_live
        selected_source = no_source
        selected_age_ms = no_age_ms
        selected_observed_at = no_observed_at
    elif selected_token:
        selected_outcome = "yes" if selected_token == yes_token else "no"
        if selected_token == yes_token:
            selected_live = yes_live
            selected_source = yes_source
            selected_age_ms = yes_age_ms
            selected_observed_at = yes_observed_at
        elif selected_token == no_token:
            selected_live = no_live
            selected_source = no_source
            selected_age_ms = no_age_ms
            selected_observed_at = no_observed_at

    entry_delta = None
    entry_delta_pct = None
    adverse_move = None
    if signal_entry is not None and selected_live is not None:
        entry_delta = selected_live - signal_entry
        if signal_entry > 0:
            entry_delta_pct = (entry_delta / signal_entry) * 100.0
        adverse_move = entry_delta > 0.0001

    model_probability = _extract_model_probability(signal, direction=direction)
    live_edge = None
    if model_probability is not None and selected_live is not None:
        live_edge = (model_probability - selected_live) * 100.0

    timing = _extract_market_timing(market_info)
    strict_sources = {"ws_strict"}

    return {
        "available": bool(selected_live is not None),
        "fetched_at": utcnow().isoformat().replace("+00:00", "Z"),
        "live_market_fetched_at": utcnow().isoformat().replace("+00:00", "Z"),
        "market_id": signal_market_id or market_id,
        "condition_id": _normalize_identifier(market_info.get("condition_id") or market_id),
        "market_question": (
            str(market_info.get("question") or getattr(signal, "market_question", "") or "").strip()
        ),
        "direction": direction,
        "selected_outcome": selected_outcome,
        "token_ids": [token for token in (yes_token, no_token) if token],
        "yes_token_id": yes_token,
        "no_token_id": no_token,
        "selected_token_id": selected_token,
        "live_yes_price": yes_live,
        "live_no_price": no_live,
        "live_selected_price": selected_live,
        "yes_price_source": yes_source,
        "no_price_source": no_source,
        "live_selected_price_source": selected_source,
        "market_data_source": selected_source,
        "source_observed_at": selected_observed_at,
        "market_data_age_ms": selected_age_ms,
        "market_data_age_seconds": ((selected_age_ms / 1000.0) if selected_age_ms is not None else None),
        "signal_entry_price": signal_entry,
        "entry_price_delta": entry_delta,
        "entry_price_delta_pct": entry_delta_pct,
        "adverse_price_move": adverse_move,
        "liquidity_usd": safe_float(market_info.get("liquidity")),
        "volume_usd": safe_float(market_info.get("volume")),
        "spread": safe_float(market_info.get("spread")),
        "model_probability": model_probability,
        "live_edge_percent": live_edge,
        "oracle_price": safe_float(market_info.get("oracle_price")),
        "oracle_source": str(market_info.get("oracle_source") or "").strip().lower() or None,
        "price_to_beat": safe_float(market_info.get("price_to_beat")),
        "oracle_prices_by_source": (
            dict(market_info.get("oracle_prices_by_source"))
            if isinstance(market_info.get("oracle_prices_by_source"), dict)
            else {}
        ),
        "oracle_age_seconds": safe_float(market_info.get("oracle_age_seconds")),
        "oracle_updated_at_ms": safe_float(market_info.get("oracle_updated_at_ms")),
        "market_end_time": timing.get("end_time"),
        "seconds_left": timing.get("seconds_left"),
        "is_live": timing.get("is_live"),
        "is_current": timing.get("is_current"),
        "market_closed": timing.get("closed"),
        "market_resolved": timing.get("resolved"),
        "history_summary": _build_history_summary(selected_history),
        "history_tail": selected_history[-20:] if selected_history else [],
    }


def _build_cached_signal_contexts(
    signal_rows: list[dict[str, Any]],
    *,
    max_history_points: int,
    strict_ws_only: bool,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    try:
        feed_manager = get_feed_manager()
    except Exception:
        return {}, signal_rows
    if feed_manager is None or not getattr(feed_manager, "_started", False):
        return {}, signal_rows

    resolved: dict[str, dict[str, Any]] = {}
    unresolved: list[dict[str, Any]] = []
    history_cache: dict[str, list[dict[str, float]]] = {}
    now_epoch = time.time()

    for row in signal_rows:
        signal = row["signal"]
        signal_id = row["signal_id"]
        market_id = row["market_lookup_id"]
        strict_ttl = _strict_ws_ttl_seconds_for_source(row.get("source"))
        market_info = dict(_extract_signal_market_hint(signal) or {})
        if not market_info:
            unresolved.append(row)
            continue
        yes_token, no_token = _extract_yes_no_tokens(market_info)
        if yes_token is None and no_token is None:
            unresolved.append(row)
            continue
        direction = row["direction"]
        selected_token = None
        if direction == "buy_yes":
            selected_token = yes_token
        elif direction == "buy_no":
            selected_token = no_token
        else:
            selected_token = yes_token or no_token
        if not selected_token:
            unresolved.append(row)
            continue

        def _cached_price(token_id: str | None) -> tuple[float | None, float | None, str | None]:
            if not token_id:
                return None, None, None
            try:
                mid = safe_float(feed_manager.cache.get_mid_price(token_id))
                age_s = safe_float(feed_manager.cache.staleness(token_id))
            except Exception:
                return None, None, None
            if mid is None:
                return None, None, None
            if age_s is not None and age_s > strict_ttl:
                if strict_ws_only:
                    return None, None, None
                return None, None, None
            observed_at_s = now_epoch - max(0.0, float(age_s or 0.0))
            return float(mid), max(0.0, float(age_s or 0.0) * 1000.0), _iso_from_epoch_seconds(observed_at_s)

        yes_live, yes_age_ms, yes_observed_at = _cached_price(yes_token)
        no_live, no_age_ms, no_observed_at = _cached_price(no_token)
        selected_live = yes_live if selected_token == yes_token else no_live
        if selected_live is None:
            unresolved.append(row)
            continue

        selected_history = history_cache.get(selected_token)
        if selected_history is None:
            selected_history = _cached_history_points_for_token(
                feed_manager,
                selected_token,
                max_history_points=max_history_points,
            )
            history_cache[selected_token] = selected_history

        resolved[signal_id] = _build_context_from_cached_market_state(
            signal,
            market_id=market_id,
            market_info=market_info,
            direction=direction,
            yes_token=yes_token,
            no_token=no_token,
            yes_live=yes_live,
            no_live=no_live,
            yes_age_ms=yes_age_ms,
            no_age_ms=no_age_ms,
            yes_observed_at=yes_observed_at,
            no_observed_at=no_observed_at,
            selected_token=selected_token,
            selected_history=selected_history,
        )

    return resolved, unresolved


async def _fetch_market_info(
    market_id: str,
    *,
    timeout_seconds: float = 3.0,
) -> Optional[dict[str, Any]]:
    normalized = _normalize_identifier(market_id)
    if not normalized:
        return None

    timeout = max(0.1, float(timeout_seconds))
    try:
        if _is_condition_id(normalized):
            return await asyncio.wait_for(
                polymarket_client.get_market_by_condition_id(normalized),
                timeout=timeout,
            )
        if _is_token_id(normalized):
            return await asyncio.wait_for(
                polymarket_client.get_market_by_token_id(normalized),
                timeout=timeout,
            )
        # Last-resort fallback: treat as condition id.
        return await asyncio.wait_for(
            polymarket_client.get_market_by_condition_id(normalized),
            timeout=timeout,
        )
    except Exception as exc:
        logger.warning(
            "Failed to fetch live market metadata",
            market_id=normalized,
            timeout_seconds=timeout,
            exc_info=exc,
        )
        return None


def _build_signal_context_rows(signals: list[Any]) -> list[dict[str, Any]]:
    signal_rows: list[dict[str, Any]] = []
    for signal in signals:
        signal_id = str(getattr(signal, "id", "") or "").strip()
        signal_market_id = _normalize_identifier(getattr(signal, "market_id", ""))
        market_hint = _extract_signal_market_hint(signal)
        hinted_condition_id = _normalize_identifier(market_hint.get("condition_id") or market_hint.get("conditionId"))
        hinted_market_id = _normalize_identifier(market_hint.get("market_id") or market_hint.get("id"))
        market_lookup_id = hinted_condition_id or hinted_market_id or signal_market_id
        if not signal_id or not market_lookup_id:
            continue
        direction = str(getattr(signal, "direction", "") or "").strip().lower()
        signal_rows.append(
            {
                "signal_id": signal_id,
                "signal_market_id": signal_market_id,
                "market_lookup_id": market_lookup_id,
                "source": str(getattr(signal, "source", "") or "").strip().lower(),
                "direction": direction,
                "market_hint": market_hint,
                "signal": signal,
            }
        )
    return signal_rows


async def build_cached_live_signal_contexts(
    signals: list[Any],
    *,
    max_history_points: int = 120,
    strict_ws_only: bool = False,
) -> dict[str, dict[str, Any]]:
    signal_rows = _build_signal_context_rows(signals)
    if not signal_rows:
        return {}
    cached_contexts, _ = _build_cached_signal_contexts(
        signal_rows,
        max_history_points=max_history_points,
        strict_ws_only=strict_ws_only,
    )
    return cached_contexts


async def build_live_signal_contexts(
    signals: list[Any],
    *,
    history_window_seconds: int = 7200,
    history_fidelity_seconds: int = 300,
    max_history_points: int = 120,
    history_tail_points: int = 20,
    max_market_concurrency: int = 12,
    max_history_concurrency: int = 24,
    market_fetch_timeout_seconds: float = 3.0,
    prices_batch_timeout_seconds: float = 3.0,
    history_fetch_timeout_seconds: float = 3.0,
    strict_ws_only: bool = False,
) -> dict[str, dict[str, Any]]:
    """Fetch live market prices + movement context for a set of trade signals."""
    signal_rows = _build_signal_context_rows(signals)
    if not signal_rows:
        return {}

    cached_contexts, unresolved_signal_rows = _build_cached_signal_contexts(
        signal_rows,
        max_history_points=max_history_points,
        strict_ws_only=strict_ws_only,
    )
    if not unresolved_signal_rows:
        return cached_contexts
    if strict_ws_only:
        # Strict execution is cache-only. If the required WS state is not
        # already hot, the caller must defer and wait for a fresh tick.
        return cached_contexts

    signal_rows = unresolved_signal_rows

    market_hints_by_lookup_id: dict[str, dict[str, Any]] = {}
    for row in signal_rows:
        market_hint = row.get("market_hint")
        market_lookup_id = row["market_lookup_id"]
        if market_hint and market_lookup_id not in market_hints_by_lookup_id:
            market_hints_by_lookup_id[market_lookup_id] = dict(market_hint)

    market_ids = sorted({row["market_lookup_id"] for row in signal_rows})
    market_infos: dict[str, dict[str, Any]] = {
        market_id: dict(market_hints_by_lookup_id.get(market_id) or {}) for market_id in market_ids
    }
    market_sem = asyncio.Semaphore(max(1, int(max_market_concurrency)))

    async def _resolve_market(market_id: str) -> None:
        async with market_sem:
            fetched = await _fetch_market_info(
                market_id,
                timeout_seconds=market_fetch_timeout_seconds,
            )
            if not isinstance(fetched, dict):
                return
            merged = dict(market_infos.get(market_id) or {})
            merged.update(fetched)
            market_infos[market_id] = merged

    market_ids_to_fetch: list[str] = []
    if not strict_ws_only:
        for market_id in market_ids:
            existing = market_infos.get(market_id) or {}
            yes_hint, no_hint = _extract_yes_no_tokens(existing)
            if yes_hint is not None or no_hint is not None:
                continue
            if _is_token_id(market_id):
                continue
            market_ids_to_fetch.append(market_id)
    await asyncio.gather(*[_resolve_market(mid) for mid in market_ids_to_fetch])

    yes_no_tokens_by_market: dict[str, tuple[Optional[str], Optional[str]]] = {}
    all_market_tokens: set[str] = set()
    for market_id in market_ids:
        info = market_infos.get(market_id) or {}
        yes_token, no_token = _extract_yes_no_tokens(info)
        # Fallback if market id itself is already a token id.
        if yes_token is None and no_token is None and _is_token_id(market_id):
            yes_token = market_id
        yes_no_tokens_by_market[market_id] = (yes_token, no_token)
        if yes_token:
            all_market_tokens.add(yes_token)
        if no_token:
            all_market_tokens.add(no_token)

    selected_tokens: set[str] = set()
    selected_token_by_signal: dict[str, Optional[str]] = {}
    max_strict_ttl_by_token: dict[str, float] = {}
    for row in signal_rows:
        signal_id = row["signal_id"]
        market_id = row["market_lookup_id"]
        direction = row["direction"]
        yes_token, no_token = yes_no_tokens_by_market.get(market_id, (None, None))
        row_strict_ttl = _strict_ws_ttl_seconds_for_source(row.get("source"))
        for token_id in (yes_token, no_token):
            normalized_token = _normalize_identifier(token_id)
            if not normalized_token:
                continue
            max_strict_ttl_by_token[normalized_token] = max(
                max_strict_ttl_by_token.get(normalized_token, 0.0),
                row_strict_ttl,
            )

        selected: Optional[str] = None
        if direction == "buy_yes":
            selected = yes_token
        elif direction == "buy_no":
            selected = no_token
        else:
            selected = yes_token or no_token

        selected_token_by_signal[signal_id] = selected
        if selected:
            selected_tokens.add(selected)

    now_epoch = time.time()
    default_strict_ttl = max(0.05, float(getattr(settings, "WS_EXECUTION_PRICE_STALE_SECONDS", 1.0) or 1.0))
    source_priority = {
        "ws_strict": 0,
        "http_batch": 1,
        "market_snapshot": 2,
        "history_tail": 3,
    }
    live_prices: dict[str, float] = {}
    live_price_meta: dict[str, dict[str, Any]] = {}
    history_points_by_token: dict[str, list[dict[str, float]]] = {}
    runtime_snapshot_tokens: set[str] = set()

    def _record_live_price(
        token_id: str,
        *,
        mid: Any,
        source: str,
        observed_at_s: float | None,
    ) -> None:
        norm = _normalize_identifier(token_id)
        if not norm:
            return
        parsed_mid = safe_float(mid)
        if parsed_mid is None or parsed_mid < 0.0 or parsed_mid > 1.01:
            return
        age_ms = None
        if observed_at_s is not None:
            age_ms = max(0.0, (now_epoch - float(observed_at_s)) * 1000.0)
        priority = source_priority.get(source, 99)
        existing = live_price_meta.get(norm)
        if isinstance(existing, dict):
            existing_age_ms = safe_float(existing.get("age_ms"), None)
            existing_priority = int(existing.get("priority", 99))
            if existing_age_ms is not None and age_ms is not None:
                if existing_age_ms + 1e-6 < age_ms:
                    return
                if age_ms + 1e-6 < existing_age_ms:
                    pass
                elif existing_priority <= priority:
                    return
            elif existing_age_ms is not None and age_ms is None:
                return
            elif existing_age_ms is None and age_ms is not None:
                pass
            elif existing_priority <= priority:
                return
        live_prices[norm] = float(parsed_mid)
        live_price_meta[norm] = {
            "source": source,
            "priority": priority,
            "observed_at_s": float(observed_at_s) if observed_at_s is not None else None,
            "observed_at": _iso_from_epoch_seconds(observed_at_s),
            "age_ms": age_ms,
        }

    for market_id in market_ids:
        info = market_infos.get(market_id) or {}
        if not bool(info.get("_runtime_market_snapshot")):
            continue
        yes_token, no_token = yes_no_tokens_by_market.get(market_id, (None, None))
        snapshot_observed_s = None
        for key in (
            "source_observed_at",
            "price_updated_at",
            "updated_at",
            "fetched_at",
            "live_market_fetched_at",
        ):
            snapshot_observed_s = _epoch_seconds_from_market_timestamp(info.get(key))
            if snapshot_observed_s is not None:
                break
        yes_snapshot = safe_float(info.get("yes_price"))
        if yes_snapshot is None:
            yes_snapshot = safe_float(info.get("up_price"))
        no_snapshot = safe_float(info.get("no_price"))
        if no_snapshot is None:
            no_snapshot = safe_float(info.get("down_price"))
        if yes_token:
            runtime_snapshot_tokens.add(yes_token)
            if yes_snapshot is not None:
                _record_live_price(
                    yes_token,
                    mid=yes_snapshot,
                    source="market_snapshot",
                    observed_at_s=snapshot_observed_s,
                )
        if no_token:
            runtime_snapshot_tokens.add(no_token)
            if no_snapshot is not None:
                _record_live_price(
                    no_token,
                    mid=no_snapshot,
                    source="market_snapshot",
                    observed_at_s=snapshot_observed_s,
                )
        history_tail = info.get("history_tail")
        normalized_history = _normalize_history_points(
            history_tail if isinstance(history_tail, list) else [],
            max_points=max_history_points,
        )
        if normalized_history and yes_token:
            history_points_by_token[yes_token] = normalized_history

    if all_market_tokens:
        token_list = sorted(all_market_tokens)
        feed_manager = None
        try:
            feed_manager = get_feed_manager()
        except Exception as exc:
            logger.warning(
                "Failed to obtain WebSocket feed manager for live pricing",
                exc_info=exc,
            )
            feed_manager = None
        polymarket_ws_tokens = [token_id for token_id in token_list if _is_token_id(token_id)]
        if feed_manager is not None and getattr(feed_manager, "_started", False):
            for token_id in token_list:
                token_norm = _normalize_identifier(token_id)
                if not token_norm:
                    continue
                strict_ttl = max_strict_ttl_by_token.get(token_norm, default_strict_ttl)
                try:
                    mid = feed_manager.cache.get_mid_price(token_norm)
                except Exception as exc:
                    logger.warning(
                        "WebSocket mid-price lookup failed",
                        token_id=token_norm,
                        exc_info=exc,
                    )
                    mid = None
                parsed_mid = safe_float(mid)
                if parsed_mid is None:
                    continue
                age_s = None
                try:
                    age_s = safe_float(feed_manager.cache.staleness(token_norm))
                except Exception as exc:
                    logger.warning(
                        "WebSocket staleness lookup failed",
                        token_id=token_norm,
                        exc_info=exc,
                    )
                    age_s = None
                if age_s is not None:
                    if age_s > strict_ttl and not strict_ws_only:
                        continue
                    source = "ws_strict"
                    observed_at_s = now_epoch - max(0.0, float(age_s))
                else:
                    try:
                        strict_fresh = feed_manager.is_fresh(token_norm, max_age_seconds=strict_ttl)
                    except Exception as exc:
                        logger.warning(
                            "WebSocket strict freshness check failed",
                            token_id=token_norm,
                            stale_seconds=strict_ttl,
                            exc_info=exc,
                        )
                        strict_fresh = False
                    if strict_fresh:
                        source = "ws_strict"
                        observed_at_s = now_epoch
                    else:
                        continue
                _record_live_price(
                    token_norm,
                    mid=parsed_mid,
                    source=source,
                    observed_at_s=observed_at_s,
                )

        unresolved = [token_id for token_id in token_list if token_id not in live_prices]
        if unresolved and not strict_ws_only:
            batch_timeout = max(0.1, float(prices_batch_timeout_seconds))
            try:
                batch = await asyncio.wait_for(
                    polymarket_client.get_prices_batch(unresolved),
                    timeout=batch_timeout,
                )
            except Exception as exc:
                logger.warning(
                    "HTTP batch price fetch failed",
                    token_count=len(unresolved),
                    timeout_seconds=batch_timeout,
                    exc_info=exc,
                )
                batch = {}
            for token_id, payload in (batch or {}).items():
                if not isinstance(payload, dict):
                    continue
                _record_live_price(
                    token_id,
                    mid=payload.get("mid"),
                    source="http_batch",
                    observed_at_s=now_epoch,
                )

    history_sem = asyncio.Semaphore(max(1, int(max_history_concurrency)))
    now_s = int(time.time())
    start_s = max(0, now_s - max(60, int(history_window_seconds)))
    fidelity = max(30, int(history_fidelity_seconds))
    history_timeout = max(0.1, float(history_fetch_timeout_seconds))
    feed_manager = None
    try:
        feed_manager = get_feed_manager()
    except Exception as exc:
        logger.warning(
            "Failed to obtain WebSocket feed manager for history lookup",
            exc_info=exc,
        )
        feed_manager = None

    async def _resolve_history(token_id: str) -> None:
        if token_id in history_points_by_token:
            return
        if feed_manager is not None and getattr(feed_manager, "_started", False):
            try:
                cached_history = feed_manager.cache.get_price_history(
                    token_id,
                    max_snapshots=max(20, int(max_history_points) * 2),
                )
            except Exception as exc:
                logger.warning(
                    "WebSocket history lookup failed",
                    token_id=token_id,
                    exc_info=exc,
                )
                cached_history = []
            normalized_cached: list[dict[str, Any]] = []
            for item in cached_history:
                if not isinstance(item, dict):
                    continue
                ts = _epoch_seconds_from_market_timestamp(item.get("timestamp"))
                mid = safe_float(item.get("mid"))
                if ts is None or mid is None:
                    continue
                normalized_cached.append({"t": ts * 1000.0, "p": float(mid)})
            normalized_from_cache = _normalize_history_points(
                normalized_cached,
                max_points=max_history_points,
            )
            if len(normalized_from_cache) >= 2:
                history_points_by_token[token_id] = normalized_from_cache
                return

        if token_id in runtime_snapshot_tokens:
            history_points_by_token[token_id] = []
            return

        if strict_ws_only:
            history_points_by_token[token_id] = []
            return

        async with history_sem:
            try:
                raw = await asyncio.wait_for(
                    polymarket_client.get_prices_history(
                        token_id,
                        start_ts=start_s,
                        end_ts=now_s,
                        fidelity=fidelity,
                    ),
                    timeout=history_timeout,
                )
            except Exception as exc:
                logger.warning(
                    "HTTP price history fetch failed",
                    token_id=token_id,
                    timeout_seconds=history_timeout,
                    exc_info=exc,
                )
                raw = []
            history_points_by_token[token_id] = _normalize_history_points(
                raw or [],
                max_points=max_history_points,
            )

    await asyncio.gather(*[_resolve_history(token_id) for token_id in sorted(selected_tokens)])

    fetched_at_dt = utcnow()
    fetched_at_iso = fetched_at_dt.isoformat().replace("+00:00", "Z")
    fetched_at_epoch = fetched_at_dt.timestamp()
    contexts: dict[str, dict[str, Any]] = dict(cached_contexts)
    for row in signal_rows:
        signal = row["signal"]
        signal_id = row["signal_id"]
        market_id = row["market_lookup_id"]
        signal_market_id = row["signal_market_id"]
        direction = row["direction"]

        market_info = market_infos.get(market_id) or {}
        yes_token, no_token = yes_no_tokens_by_market.get(market_id, (None, None))
        selected_token = selected_token_by_signal.get(signal_id)

        yes_norm = _normalize_identifier(yes_token)
        no_norm = _normalize_identifier(no_token)
        yes_live = live_prices.get(yes_norm) if yes_norm else None
        no_live = live_prices.get(no_norm) if no_norm else None
        yes_meta = dict(live_price_meta.get(yes_norm) or {})
        no_meta = dict(live_price_meta.get(no_norm) or {})
        snapshot_observed_s = None
        for key in (
            "source_observed_at",
            "price_updated_at",
            "updated_at",
            "fetched_at",
            "live_market_fetched_at",
        ):
            snapshot_observed_s = _epoch_seconds_from_market_timestamp(market_info.get(key))
            if snapshot_observed_s is not None:
                break
        if yes_live is None and not strict_ws_only:
            yes_snapshot = safe_float(market_info.get("yes_price"))
            if yes_snapshot is not None and 0.0 <= yes_snapshot <= 1.01:
                yes_live = float(yes_snapshot)
                yes_meta = {
                    "source": "market_snapshot",
                    "observed_at_s": snapshot_observed_s,
                    "observed_at": _iso_from_epoch_seconds(snapshot_observed_s),
                    "age_ms": (
                        max(0.0, (now_epoch - snapshot_observed_s) * 1000.0)
                        if snapshot_observed_s is not None
                        else None
                    ),
                }
        if no_live is None and not strict_ws_only:
            no_snapshot = safe_float(market_info.get("no_price"))
            if no_snapshot is not None and 0.0 <= no_snapshot <= 1.01:
                no_live = float(no_snapshot)
                no_meta = {
                    "source": "market_snapshot",
                    "observed_at_s": snapshot_observed_s,
                    "observed_at": _iso_from_epoch_seconds(snapshot_observed_s),
                    "age_ms": (
                        max(0.0, (now_epoch - snapshot_observed_s) * 1000.0)
                        if snapshot_observed_s is not None
                        else None
                    ),
                }

        selected_outcome: Optional[str] = None
        selected_live: Optional[float] = None
        selected_meta: dict[str, Any] = {}
        if direction == "buy_yes":
            selected_outcome = "yes"
            selected_live = yes_live
            selected_meta = yes_meta
        elif direction == "buy_no":
            selected_outcome = "no"
            selected_live = no_live
            selected_meta = no_meta
        elif selected_token:
            selected_outcome = "yes" if selected_token == yes_token else "no"
            if selected_token == yes_token:
                selected_live = yes_live
                selected_meta = yes_meta
            elif selected_token == no_token:
                selected_live = no_live
                selected_meta = no_meta

        selected_history = history_points_by_token.get(selected_token, []) if selected_token else []
        if selected_live is None and selected_history and not strict_ws_only:
            selected_live = selected_history[-1]["p"]
            selected_history_ts_s = safe_float(selected_history[-1].get("t"))
            if selected_history_ts_s is not None:
                selected_history_ts_s = selected_history_ts_s / 1000.0
            selected_meta = {
                "source": "history_tail",
                "observed_at_s": selected_history_ts_s,
                "observed_at": _iso_from_epoch_seconds(selected_history_ts_s),
                "age_ms": (
                    max(0.0, (now_epoch - selected_history_ts_s) * 1000.0)
                    if selected_history_ts_s is not None
                    else None
                ),
            }

        signal_entry = safe_float(getattr(signal, "entry_price", None))
        entry_delta = None
        entry_delta_pct = None
        adverse_move = None
        if signal_entry is not None and selected_live is not None:
            entry_delta = selected_live - signal_entry
            if signal_entry > 0:
                entry_delta_pct = (entry_delta / signal_entry) * 100.0
            # For a buyer, paying more than original signal entry is adverse.
            adverse_move = entry_delta > 0.0001

        model_probability = _extract_model_probability(signal, direction=direction)
        live_edge = None
        if model_probability is not None and selected_live is not None:
            live_edge = (model_probability - selected_live) * 100.0

        timing = _extract_market_timing(market_info)
        selected_source = str(selected_meta.get("source") or "").strip().lower()
        strict_sources = {"ws_strict"}
        if strict_ws_only and selected_source not in strict_sources:
            selected_live = None
            selected_meta = {}
            selected_source = ""
        selected_age_ms = safe_float(selected_meta.get("age_ms"))
        selected_observed_at = selected_meta.get("observed_at") or None
        if selected_observed_at is None and selected_live is not None and selected_source == "http_batch":
            selected_observed_at = fetched_at_iso
        if selected_age_ms is None and selected_observed_at is not None:
            selected_observed_s = _epoch_seconds_from_market_timestamp(selected_observed_at)
            if selected_observed_s is not None:
                selected_age_ms = max(0.0, (fetched_at_epoch - selected_observed_s) * 1000.0)
        yes_source = str(yes_meta.get("source") or "").strip().lower() or None
        no_source = str(no_meta.get("source") or "").strip().lower() or None
        contexts[signal_id] = {
            "available": bool(selected_live is not None),
            "fetched_at": fetched_at_iso,
            "live_market_fetched_at": fetched_at_iso,
            "market_id": signal_market_id or market_id,
            "condition_id": _normalize_identifier(market_info.get("condition_id") or market_id),
            "market_question": (
                str(market_info.get("question") or getattr(signal, "market_question", "") or "").strip()
            ),
            "direction": direction,
            "selected_outcome": selected_outcome,
            "token_ids": [token for token in (yes_token, no_token) if token],
            "yes_token_id": yes_token,
            "no_token_id": no_token,
            "selected_token_id": selected_token,
            "live_yes_price": yes_live,
            "live_no_price": no_live,
            "live_selected_price": selected_live,
            "yes_price_source": yes_source,
            "no_price_source": no_source,
            "live_selected_price_source": selected_source or None,
            "market_data_source": selected_source or None,
            "source_observed_at": selected_observed_at,
            "market_data_age_ms": selected_age_ms,
            "market_data_age_seconds": ((selected_age_ms / 1000.0) if selected_age_ms is not None else None),
            "signal_entry_price": signal_entry,
            "entry_price_delta": entry_delta,
            "entry_price_delta_pct": entry_delta_pct,
            "adverse_price_move": adverse_move,
            "liquidity_usd": safe_float(market_info.get("liquidity")),
            "volume_usd": safe_float(market_info.get("volume")),
            "spread": safe_float(market_info.get("spread")),
            "model_probability": model_probability,
            "live_edge_percent": live_edge,
            "oracle_price": safe_float(market_info.get("oracle_price")),
            "oracle_source": str(market_info.get("oracle_source") or "").strip().lower() or None,
            "price_to_beat": safe_float(market_info.get("price_to_beat")),
            "oracle_prices_by_source": (
                dict(market_info.get("oracle_prices_by_source"))
                if isinstance(market_info.get("oracle_prices_by_source"), dict)
                else {}
            ),
            "oracle_age_seconds": safe_float(market_info.get("oracle_age_seconds")),
            "oracle_updated_at_ms": safe_float(market_info.get("oracle_updated_at_ms")),
            "market_end_time": timing.get("end_time"),
            "seconds_left": timing.get("seconds_left"),
            "is_live": timing.get("is_live"),
            "is_current": timing.get("is_current"),
            "market_closed": timing.get("closed"),
            "market_resolved": timing.get("resolved"),
            "history_summary": _build_history_summary(selected_history),
            "history_tail": (selected_history[-max(1, int(history_tail_points)) :] if selected_history else []),
        }
    return contexts


class RuntimeTradeSignalView:
    """Read-only signal wrapper with runtime live-market overrides."""

    def __init__(
        self,
        base_signal: Any,
        *,
        live_context: Optional[dict[str, Any]] = None,
    ) -> None:
        self._base_signal = base_signal
        self.live_context = live_context or {}

        live_entry = safe_float(self.live_context.get("live_selected_price"))
        live_edge = safe_float(self.live_context.get("live_edge_percent"))
        self.entry_price = live_entry if live_entry is not None else getattr(base_signal, "entry_price", None)
        self.edge_percent = live_edge if live_edge is not None else getattr(base_signal, "edge_percent", None)

        payload = getattr(base_signal, "payload_json", None)
        if not isinstance(payload, dict):
            payload = {}
        if self.live_context:
            payload = {
                **payload,
                "live_market": self.live_context,
            }
        self.payload_json = payload

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base_signal, name)
