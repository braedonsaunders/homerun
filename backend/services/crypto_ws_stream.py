from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from config import settings
from services.redis_streams import redis_streams


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _normalize_token_ids(raw_ids: Any, *, limit: int = 5000) -> list[str]:
    if not isinstance(raw_ids, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for raw in raw_ids:
        token_id = str(raw or "").strip()
        if not token_id or token_id in seen:
            continue
        seen.add(token_id)
        out.append(token_id)
        if len(out) >= max(1, int(limit)):
            break
    return out


def _normalize_source(value: Any) -> str:
    return str(value or "").strip().lower()


async def publish_crypto_ws_update_batch(
    *,
    token_ids: list[str],
    source: str | None,
    emitted_at: str | None = None,
) -> str | None:
    normalized_ids = _normalize_token_ids(token_ids)
    if not normalized_ids:
        return None
    payload = {
        "source": _normalize_source(source),
        "token_ids": normalized_ids,
        "token_count": int(len(normalized_ids)),
        "emitted_at": str(emitted_at or _utc_now_iso()),
    }
    return await redis_streams.append_json(
        str(settings.CRYPTO_WS_UPDATE_STREAM_KEY),
        payload,
        maxlen=max(1000, int(settings.CRYPTO_WS_UPDATE_STREAM_MAXLEN)),
    )


async def ensure_crypto_ws_update_group() -> bool:
    return await redis_streams.ensure_consumer_group(
        stream=str(settings.CRYPTO_WS_UPDATE_STREAM_KEY),
        group=str(settings.CRYPTO_WS_UPDATE_STREAM_GROUP),
        start_id="$",
        create_stream=True,
    )


def _normalize_batch_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    token_ids = _normalize_token_ids(payload.get("token_ids"))
    if not token_ids:
        return None
    return {
        "source": _normalize_source(payload.get("source")),
        "token_ids": token_ids,
        "token_count": int(len(token_ids)),
        "emitted_at": payload.get("emitted_at"),
    }


async def read_crypto_ws_update_batches(
    *,
    consumer: str,
    block_ms: int | None = None,
    count: int | None = None,
    include_pending: bool = False,
) -> list[tuple[str, dict[str, Any]]]:
    rows = await redis_streams.read_group_raw(
        stream=str(settings.CRYPTO_WS_UPDATE_STREAM_KEY),
        group=str(settings.CRYPTO_WS_UPDATE_STREAM_GROUP),
        consumer=str(consumer),
        block_ms=(
            max(1, int(block_ms))
            if block_ms is not None
            else max(1, int(settings.CRYPTO_WS_UPDATE_STREAM_BLOCK_MS))
        ),
        count=(
            max(1, int(count))
            if count is not None
            else max(1, int(settings.CRYPTO_WS_UPDATE_STREAM_READ_COUNT))
        ),
        include_pending=bool(include_pending),
    )
    out: list[tuple[str, dict[str, Any]]] = []
    for entry_id, raw_payload in rows:
        try:
            payload = json.loads(raw_payload)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        normalized = _normalize_batch_payload(payload)
        if normalized is None:
            continue
        out.append((str(entry_id), normalized))
    return out


async def auto_claim_crypto_ws_update_batches(
    *,
    consumer: str,
    min_idle_ms: int | None = None,
    start_id: str = "0-0",
    count: int | None = None,
) -> tuple[str, list[tuple[str, dict[str, Any]]]]:
    next_start_id, rows = await redis_streams.auto_claim_raw(
        stream=str(settings.CRYPTO_WS_UPDATE_STREAM_KEY),
        group=str(settings.CRYPTO_WS_UPDATE_STREAM_GROUP),
        consumer=str(consumer),
        min_idle_ms=(
            max(1, int(min_idle_ms))
            if min_idle_ms is not None
            else max(1, int(settings.CRYPTO_WS_UPDATE_STREAM_CLAIM_IDLE_MS))
        ),
        start_id=str(start_id or "0-0"),
        count=(
            max(1, int(count))
            if count is not None
            else max(1, int(settings.CRYPTO_WS_UPDATE_STREAM_CLAIM_READ_COUNT))
        ),
    )
    out: list[tuple[str, dict[str, Any]]] = []
    for entry_id, raw_payload in rows:
        try:
            payload = json.loads(raw_payload)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        normalized = _normalize_batch_payload(payload)
        if normalized is None:
            continue
        out.append((str(entry_id), normalized))
    return str(next_start_id or start_id), out


async def ack_crypto_ws_update_batches(entry_ids: list[str]) -> int:
    return await redis_streams.ack(
        stream=str(settings.CRYPTO_WS_UPDATE_STREAM_KEY),
        group=str(settings.CRYPTO_WS_UPDATE_STREAM_GROUP),
        entry_ids=entry_ids,
    )
