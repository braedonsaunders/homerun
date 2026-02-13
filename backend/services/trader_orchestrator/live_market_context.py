from __future__ import annotations

import asyncio
import re
import time
from typing import Any, Optional

from services.polymarket import polymarket_client
from utils.utcnow import utcnow

_POLYMARKET_CONDITION_ID_RE = re.compile(r"^0x[0-9a-f]{64}$")
_POLYMARKET_NUMERIC_TOKEN_ID_RE = re.compile(r"^\d{18,}$")
_POLYMARKET_HEX_TOKEN_ID_RE = re.compile(r"^(?:0x)?[0-9a-f]{40,}$")


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _normalize_identifier(value: Any) -> str:
    return str(value or "").strip().lower()


def _is_condition_id(value: str) -> bool:
    return bool(_POLYMARKET_CONDITION_ID_RE.fullmatch(value))


def _is_token_id(value: str) -> bool:
    if _is_condition_id(value):
        return False
    return bool(
        _POLYMARKET_NUMERIC_TOKEN_ID_RE.fullmatch(value)
        or _POLYMARKET_HEX_TOKEN_ID_RE.fullmatch(value)
    )


def _extract_token_ids(market_info: dict[str, Any]) -> list[str]:
    raw = (
        market_info.get("token_ids")
        or market_info.get("clob_token_ids")
        or market_info.get("tokenIds")
        or []
    )
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


def _normalize_history_points(
    points: list[dict[str, Any]],
    *,
    max_points: int,
) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    for item in points:
        if not isinstance(item, dict):
            continue
        t = _safe_float(item.get("t"))
        p = _safe_float(item.get("p"))
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
    entry = _safe_float(getattr(signal, "entry_price", None))
    edge = _safe_float(getattr(signal, "edge_percent", None))
    if entry is not None and edge is not None:
        implied = entry + (edge / 100.0)
        if 0.0 <= implied <= 1.0:
            return implied

    payload = getattr(signal, "payload_json", None)
    if not isinstance(payload, dict):
        return None

    for key in ("model_probability", "selected_probability"):
        value = _safe_float(payload.get(key))
        if value is not None and 0.0 <= value <= 1.0:
            return value

    # Common field for directional intents.
    value = _safe_float(payload.get("expected_payout"))
    if value is not None and 0.0 <= value <= 1.0:
        return value

    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return None

    weather = metadata.get("weather")
    if isinstance(weather, dict):
        consensus_yes = _safe_float(weather.get("consensus_probability"))
        if consensus_yes is not None and 0.0 <= consensus_yes <= 1.0:
            return (1.0 - consensus_yes) if direction == "buy_no" else consensus_yes

    for key in ("model_probability", "selected_probability"):
        value = _safe_float(metadata.get(key))
        if value is not None and 0.0 <= value <= 1.0:
            return value

    return None


async def _fetch_market_info(market_id: str) -> Optional[dict[str, Any]]:
    normalized = _normalize_identifier(market_id)
    if not normalized:
        return None

    try:
        if _is_condition_id(normalized):
            return await polymarket_client.get_market_by_condition_id(normalized)
        if _is_token_id(normalized):
            return await polymarket_client.get_market_by_token_id(normalized)
        # Last-resort fallback: treat as condition id.
        return await polymarket_client.get_market_by_condition_id(normalized)
    except Exception:
        return None


async def build_live_signal_contexts(
    signals: list[Any],
    *,
    history_window_seconds: int = 7200,
    history_fidelity_seconds: int = 300,
    max_history_points: int = 120,
    history_tail_points: int = 20,
    max_market_concurrency: int = 12,
    max_history_concurrency: int = 24,
) -> dict[str, dict[str, Any]]:
    """Fetch live market prices + movement context for a set of trade signals."""
    signal_rows: list[dict[str, Any]] = []
    for signal in signals:
        signal_id = str(getattr(signal, "id", "") or "").strip()
        market_id = _normalize_identifier(getattr(signal, "market_id", ""))
        if not signal_id or not market_id:
            continue
        direction = str(getattr(signal, "direction", "") or "").strip().lower()
        signal_rows.append(
            {
                "signal_id": signal_id,
                "market_id": market_id,
                "direction": direction,
                "signal": signal,
            }
        )
    if not signal_rows:
        return {}

    market_ids = sorted({row["market_id"] for row in signal_rows})
    market_infos: dict[str, Optional[dict[str, Any]]] = {}
    market_sem = asyncio.Semaphore(max(1, int(max_market_concurrency)))

    async def _resolve_market(market_id: str) -> None:
        async with market_sem:
            market_infos[market_id] = await _fetch_market_info(market_id)

    await asyncio.gather(*[_resolve_market(mid) for mid in market_ids])

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

    live_prices: dict[str, float] = {}
    if all_market_tokens:
        try:
            batch = await polymarket_client.get_prices_batch(sorted(all_market_tokens))
            for token_id, payload in (batch or {}).items():
                norm = _normalize_identifier(token_id)
                if not norm or not isinstance(payload, dict):
                    continue
                mid = _safe_float(payload.get("mid"))
                if mid is None:
                    continue
                if 0.0 <= mid <= 1.0:
                    live_prices[norm] = mid
        except Exception:
            live_prices = {}

    selected_tokens: set[str] = set()
    selected_token_by_signal: dict[str, Optional[str]] = {}
    for row in signal_rows:
        signal_id = row["signal_id"]
        market_id = row["market_id"]
        direction = row["direction"]
        yes_token, no_token = yes_no_tokens_by_market.get(market_id, (None, None))

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

    history_points_by_token: dict[str, list[dict[str, float]]] = {}
    history_sem = asyncio.Semaphore(max(1, int(max_history_concurrency)))
    now_s = int(time.time())
    start_s = max(0, now_s - max(60, int(history_window_seconds)))
    fidelity = max(30, int(history_fidelity_seconds))

    async def _resolve_history(token_id: str) -> None:
        async with history_sem:
            try:
                raw = await polymarket_client.get_prices_history(
                    token_id,
                    start_ts=start_s,
                    end_ts=now_s,
                    fidelity=fidelity,
                )
            except Exception:
                raw = []
            history_points_by_token[token_id] = _normalize_history_points(
                raw or [],
                max_points=max_history_points,
            )

    await asyncio.gather(*[_resolve_history(token_id) for token_id in sorted(selected_tokens)])

    fetched_at_iso = utcnow().isoformat() + "Z"
    contexts: dict[str, dict[str, Any]] = {}
    for row in signal_rows:
        signal = row["signal"]
        signal_id = row["signal_id"]
        market_id = row["market_id"]
        direction = row["direction"]

        market_info = market_infos.get(market_id) or {}
        yes_token, no_token = yes_no_tokens_by_market.get(market_id, (None, None))
        selected_token = selected_token_by_signal.get(signal_id)

        yes_live = (
            live_prices.get(yes_token) if yes_token else None
        )
        no_live = (
            live_prices.get(no_token) if no_token else None
        )
        if yes_live is None:
            yes_live = _safe_float(market_info.get("yes_price"))
        if no_live is None:
            no_live = _safe_float(market_info.get("no_price"))

        selected_outcome: Optional[str] = None
        selected_live: Optional[float] = None
        if direction == "buy_yes":
            selected_outcome = "yes"
            selected_live = yes_live
        elif direction == "buy_no":
            selected_outcome = "no"
            selected_live = no_live
        elif selected_token:
            selected_outcome = "yes" if selected_token == yes_token else "no"
            if selected_token == yes_token:
                selected_live = yes_live
            elif selected_token == no_token:
                selected_live = no_live

        selected_history = (
            history_points_by_token.get(selected_token, []) if selected_token else []
        )
        if selected_live is None and selected_history:
            selected_live = selected_history[-1]["p"]

        signal_entry = _safe_float(getattr(signal, "entry_price", None))
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

        contexts[signal_id] = {
            "available": bool(selected_live is not None),
            "fetched_at": fetched_at_iso,
            "market_id": market_id,
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
            "signal_entry_price": signal_entry,
            "entry_price_delta": entry_delta,
            "entry_price_delta_pct": entry_delta_pct,
            "adverse_price_move": adverse_move,
            "model_probability": model_probability,
            "live_edge_percent": live_edge,
            "history_summary": _build_history_summary(selected_history),
            "history_tail": (
                selected_history[-max(1, int(history_tail_points)) :]
                if selected_history
                else []
            ),
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

        live_entry = _safe_float(self.live_context.get("live_selected_price"))
        live_edge = _safe_float(self.live_context.get("live_edge_percent"))
        self.entry_price = (
            live_entry if live_entry is not None else getattr(base_signal, "entry_price", None)
        )
        self.edge_percent = (
            live_edge if live_edge is not None else getattr(base_signal, "edge_percent", None)
        )

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
