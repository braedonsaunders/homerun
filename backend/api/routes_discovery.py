from collections import defaultdict
from datetime import datetime, timedelta
import json
from utils.utcnow import utcnow
import uuid
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import func, select, delete
from typing import Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import (
    DiscoveredWallet,
    ScannerSnapshot,
    TrackedWallet,
    TraderGroup,
    TraderGroupMember,
    WalletActivityRollup,
    WalletCluster,
    get_db_session,
)
from services import discovery_shared_state
from services.insider_detector import insider_detector
from services.pause_state import global_pause_state
from services.wallet_discovery import wallet_discovery
from services.wallet_intelligence import wallet_intelligence
from services.smart_wallet_pool import (
    smart_wallet_pool,
    POOL_FLAG_MANUAL_EXCLUDE,
    POOL_FLAG_MANUAL_INCLUDE,
    POOL_FLAG_BLACKLISTED,
    POOL_RECOMPUTE_MODE_QUALITY_ONLY,
    POOL_RECOMPUTE_MODE_BALANCED,
)
from services.wallet_tracker import wallet_tracker
from services.worker_state import read_worker_snapshot
from services.polymarket import polymarket_client
from utils.validation import validate_eth_address

# Maps time_period query values to rolling window keys stored in the DB
TIME_PERIOD_TO_WINDOW_KEY = {
    "24h": "1d",
    "7d": "7d",
    "30d": "30d",
    "90d": "90d",
}

discovery_router = APIRouter()


GROUP_SOURCE_TYPES = {
    "manual",
    "suggested_cluster",
    "suggested_tag",
    "suggested_pool",
}


async def _build_discovery_status(session: AsyncSession) -> dict:
    stats = await wallet_discovery.get_discovery_stats()
    worker_status = await discovery_shared_state.get_discovery_status_from_db(session)

    stats["is_running"] = bool(worker_status.get("running", stats.get("is_running", False)))
    stats["last_run_at"] = worker_status.get("last_run_at") or stats.get("last_run_at")
    stats["wallets_discovered_last_run"] = int(
        worker_status.get(
            "wallets_discovered_last_run",
            stats.get("wallets_discovered_last_run", 0),
        )
    )
    stats["wallets_analyzed_last_run"] = int(
        worker_status.get(
            "wallets_analyzed_last_run",
            stats.get("wallets_analyzed_last_run", 0),
        )
    )
    stats["current_activity"] = worker_status.get("current_activity")
    stats["interval_minutes"] = int(worker_status.get("run_interval_minutes", 60))
    stats["paused"] = bool(worker_status.get("paused", False))
    stats["priority_backlog_mode"] = bool(worker_status.get("priority_backlog_mode", True))
    stats["requested_run_at"] = worker_status.get("requested_run_at")
    return stats


class CreateTraderGroupRequest(BaseModel):
    name: str = Field(min_length=2, max_length=80)
    description: Optional[str] = Field(default=None, max_length=500)
    wallet_addresses: list[str] = Field(default_factory=list)
    source_type: str = Field(default="manual")
    suggestion_key: Optional[str] = Field(default=None, max_length=200)
    criteria: dict = Field(default_factory=dict)
    auto_track_members: bool = True
    source_label: str = Field(default="manual", max_length=40)


class UpdateTraderGroupMembersRequest(BaseModel):
    wallet_addresses: list[str] = Field(default_factory=list)
    add_to_tracking: bool = True
    source_label: str = Field(default="manual", max_length=40)


class PoolFlagRequest(BaseModel):
    reason: Optional[str] = Field(default=None, max_length=240)


def _normalize_wallet_addresses(addresses: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in addresses:
        if not raw:
            continue
        addr = validate_eth_address(str(raw).strip()).lower()
        if addr in seen:
            continue
        normalized.append(addr)
        seen.add(addr)
    return normalized


def _suggestion_id(kind: str, key: str) -> str:
    safe_key = "".join(ch if ch.isalnum() else "_" for ch in key.lower())
    return f"{kind}:{safe_key}"


def _coerce_source_flags(raw: object) -> dict:
    if isinstance(raw, dict):
        return dict(raw)
    return {}


def _pool_flags(source_flags: dict) -> dict[str, bool]:
    return {
        "manual_include": bool(source_flags.get(POOL_FLAG_MANUAL_INCLUDE)),
        "manual_exclude": bool(source_flags.get(POOL_FLAG_MANUAL_EXCLUDE)),
        "blacklisted": bool(source_flags.get(POOL_FLAG_BLACKLISTED)),
    }


def _market_categories_from_flags(source_flags: dict) -> list[str]:
    categories = [
        key.replace("leaderboard_category_", "")
        for key, enabled in source_flags.items()
        if key.startswith("leaderboard_category_") and bool(enabled)
    ]
    return sorted(set(categories))


def _normalize_win_rate_ratio(value: object) -> float:
    """Normalize win-rate values to ratio scale [0, 1].

    Discovery historically stores ratios (0.0-1.0), but legacy payloads may
    contain percent values (0.0-100.0). Keep API sorting/filtering resilient.
    """
    try:
        wr = float(value or 0.0)
    except Exception:
        return 0.0
    if wr > 1.0:
        wr = wr / 100.0
    return max(0.0, min(wr, 1.0))


def _normalize_market_id(value: object) -> str:
    return str(value or "").strip().lower()


def _is_placeholder_market_question(question: object) -> bool:
    text = str(question or "").strip().lower()
    return text.startswith("market 0x")


def _humanize_market_slug(slug: object) -> str:
    raw = str(slug or "").strip().replace("_", "-")
    if not raw:
        return ""
    parts = [p for p in raw.split("-") if p]
    if not parts:
        return ""
    return " ".join(p[:1].upper() + p[1:] for p in parts)


def _extract_yes_no_from_history(history: list[dict]) -> tuple[Optional[float], Optional[float]]:
    if not history:
        return None, None
    last = history[-1] if isinstance(history[-1], dict) else {}
    yes = last.get("yes")
    no = last.get("no")
    try:
        yes_v = float(yes) if yes is not None else None
    except Exception:
        yes_v = None
    try:
        no_v = float(no) if no is not None else None
    except Exception:
        no_v = None
    return yes_v, no_v


def _extract_outcome_labels(raw: object) -> list[str]:
    candidates: list[object] = []
    if isinstance(raw, list):
        candidates = raw
    elif isinstance(raw, tuple):
        candidates = list(raw)
    elif isinstance(raw, str):
        text = raw.strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
            except Exception:
                parsed = None
            if isinstance(parsed, list):
                candidates = parsed

    labels: list[str] = []
    for item in candidates:
        text = str(item or "").strip()
        if text:
            labels.append(text)
    return labels[:2]


def _extract_outcome_prices(raw: object) -> tuple[Optional[float], Optional[float]]:
    candidates: list[object] = []
    if isinstance(raw, list):
        candidates = raw
    elif isinstance(raw, tuple):
        candidates = list(raw)
    elif isinstance(raw, str):
        text = raw.strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
            except Exception:
                parsed = None
            if isinstance(parsed, list):
                candidates = parsed

    parsed_values: list[float] = []
    for item in candidates:
        value = _safe_float(item)
        if value is not None:
            parsed_values.append(value)
    if len(parsed_values) >= 2:
        return parsed_values[0], parsed_values[1]
    if len(parsed_values) == 1:
        return parsed_values[0], None
    return None, None


async def _load_scanner_market_history(session: AsyncSession) -> dict[str, list[dict]]:
    """Read scanner snapshot market history map used by main opportunities sparklines."""
    result = await session.execute(select(ScannerSnapshot).where(ScannerSnapshot.id == "latest"))
    row = result.scalar_one_or_none()
    if row is None or not isinstance(row.market_history_json, dict):
        return {}

    out: dict[str, list[dict]] = {}
    for market_id, points in row.market_history_json.items():
        key = _normalize_market_id(market_id)
        if not key or not isinstance(points, list):
            continue
        out[key] = [p for p in points if isinstance(p, dict)]
    return out


def _attach_market_history_to_signal_rows(
    rows: list[dict],
    market_history: dict[str, list[dict]],
) -> None:
    """Attach sparkline payload and latest YES/NO prices to confluence/insider rows."""
    if not rows:
        return

    for row in rows:
        market_id = _normalize_market_id(row.get("market_id"))
        if not market_id:
            continue

        history = market_history.get(market_id, [])
        if len(history) >= 2:
            # Keep payload compact and consistent with AI context-pack limits.
            compact = history[-20:]
            row["price_history"] = compact
            yes_price, no_price = _extract_yes_no_from_history(compact)
            if yes_price is not None:
                row["yes_price"] = yes_price
            if no_price is not None:
                row["no_price"] = no_price

        if _is_placeholder_market_question(row.get("market_question")):
            slug_fallback = _humanize_market_slug(row.get("market_slug"))
            if slug_fallback:
                row["market_question"] = slug_fallback


def _safe_float(value: object) -> Optional[float]:
    try:
        parsed = float(value)
    except Exception:
        return None
    if parsed != parsed or parsed in (float("inf"), float("-inf")):
        return None
    return parsed


def _normalize_signal_wallet_address(value: object) -> Optional[str]:
    text = str(value or "").strip().lower()
    if not text or not text.startswith("0x"):
        return None
    return text


def _extract_signal_wallet_addresses(signal: dict[str, Any]) -> list[str]:
    addresses: set[str] = set()

    def _add_candidate(raw: object) -> None:
        normalized = _normalize_signal_wallet_address(raw)
        if normalized:
            addresses.add(normalized)

    raw_wallets = signal.get("wallets")
    if isinstance(raw_wallets, list):
        for item in raw_wallets:
            if isinstance(item, dict):
                _add_candidate(item.get("address"))
            else:
                _add_candidate(item)

    top_wallet = signal.get("top_wallet")
    if isinstance(top_wallet, dict):
        _add_candidate(top_wallet.get("address"))

    top_wallets = signal.get("top_wallets")
    if isinstance(top_wallets, list):
        for item in top_wallets:
            if isinstance(item, dict):
                _add_candidate(item.get("address"))

    wallet_addresses = signal.get("wallet_addresses")
    if isinstance(wallet_addresses, list):
        for item in wallet_addresses:
            _add_candidate(item)

    return sorted(addresses)


def _signal_has_direction(signal: dict[str, Any]) -> bool:
    outcome = str(signal.get("outcome") or "").strip().upper()
    if outcome in {"YES", "NO"}:
        return True

    direction = str(signal.get("direction") or "").strip().lower()
    if direction in {"buy_yes", "buy_no", "buy", "sell", "yes", "no"}:
        return True

    signal_type = str(signal.get("signal_type") or "").strip().lower()
    return any(token in signal_type for token in ("buy", "sell", "accumulation"))


def _signal_has_price_reference(signal: dict[str, Any]) -> tuple[bool, bool]:
    prices: list[float] = []
    for key in ("yes_price", "no_price", "entry_price", "avg_entry_price"):
        parsed = _safe_float(signal.get(key))
        if parsed is not None:
            prices.append(parsed)

    if not prices:
        return False, False
    return True, all(0.0 <= value <= 1.0 for value in prices)


def _build_signal_validation_payload(
    signal: dict[str, Any],
    wallet_addresses: list[str],
    has_qualified_source: bool,
) -> dict[str, Any]:
    has_market_id = bool(_normalize_market_id(signal.get("market_id")))
    has_wallets = bool(wallet_addresses)
    has_direction = _signal_has_direction(signal)
    has_price_reference, price_in_bounds = _signal_has_price_reference(signal)
    upstream_tradable = bool(signal.get("is_tradeable", True))

    reasons: list[str] = []
    if not has_market_id:
        reasons.append("missing_market_id")
    if not has_wallets:
        reasons.append("missing_wallets")
    if not has_direction:
        reasons.append("missing_direction")
    if not has_price_reference:
        reasons.append("missing_price_reference")
    elif not price_in_bounds:
        reasons.append("price_out_of_bounds")
    if not has_qualified_source:
        reasons.append("unqualified_wallet_source")
    if not upstream_tradable:
        reasons.append("market_not_tradable")

    is_valid = has_market_id and has_wallets and has_direction and has_price_reference and price_in_bounds
    is_actionable = is_valid and has_qualified_source
    is_tradeable = upstream_tradable and is_actionable

    return {
        "is_valid": is_valid,
        "is_actionable": is_actionable,
        "is_tradeable": is_tradeable,
        "checks": {
            "has_market_id": has_market_id,
            "has_wallets": has_wallets,
            "has_direction": has_direction,
            "has_price_reference": has_price_reference,
            "price_in_bounds": price_in_bounds,
            "has_qualified_source": has_qualified_source,
            "upstream_tradable": upstream_tradable,
        },
        "reasons": reasons,
    }


def _extract_outcome_labels_from_market_info(
    market_info: Optional[dict[str, Any]],
    *,
    fallback_slug: str = "",
    fallback_question: str = "",
) -> list[str]:
    if not isinstance(market_info, dict):
        market_info = {}

    labels = _extract_outcome_labels(market_info.get("outcomes"))
    if not labels:
        tokens = market_info.get("tokens")
        if isinstance(tokens, list):
            from_tokens: list[str] = []
            for token in tokens:
                if not isinstance(token, dict):
                    continue
                label = str(token.get("outcome") or token.get("name") or "").strip()
                if label:
                    from_tokens.append(label)
                if len(from_tokens) >= 2:
                    break
            labels = from_tokens

    if not labels:
        slug = str(market_info.get("slug") or fallback_slug or "").strip().lower()
        if slug:
            parts = [part for part in slug.split("-") if part]
            if len(parts) >= 3 and 2 <= len(parts[1]) <= 4 and 2 <= len(parts[2]) <= 4:
                if parts[1].isalpha() and parts[2].isalpha():
                    labels = [parts[1].upper(), parts[2].upper()]

    if not labels:
        question = str(
            market_info.get("groupItemTitle") or market_info.get("question") or fallback_question or ""
        ).strip()
        if " vs " in question.lower():
            lowered = question.lower()
            splitter = " vs. " if " vs. " in lowered else " vs "
            chunks = [chunk.strip() for chunk in question.split(splitter) if chunk.strip()]
            if len(chunks) >= 2:
                labels = [chunks[0], chunks[1]]

    if not labels:
        return ["Yes", "No"]
    if len(labels) == 1:
        return [labels[0], "No"]
    return labels[:2]


def _extract_outcome_prices_from_market_info(
    market_info: Optional[dict[str, Any]],
) -> tuple[Optional[float], Optional[float]]:
    if not isinstance(market_info, dict):
        return None, None

    yes_price = _safe_float(
        market_info.get("yes_price") if market_info.get("yes_price") is not None else market_info.get("yesPrice")
    )
    no_price = _safe_float(
        market_info.get("no_price") if market_info.get("no_price") is not None else market_info.get("noPrice")
    )
    if yes_price is not None or no_price is not None:
        return yes_price, no_price

    prices = market_info.get("outcome_prices")
    if prices is None:
        prices = market_info.get("outcomePrices")
    return _extract_outcome_prices(prices)


async def _attach_signal_market_metadata(rows: list[dict]) -> list[dict]:
    if not rows:
        return rows

    market_info_cache: dict[str, Optional[dict[str, Any]]] = {}

    async def _load_market_info(market_id: str) -> Optional[dict[str, Any]]:
        normalized = _normalize_market_id(market_id)
        if not normalized:
            return None
        if normalized in market_info_cache:
            return market_info_cache[normalized]
        info: Optional[dict[str, Any]] = None
        try:
            if normalized.startswith("0x"):
                info = await polymarket_client.get_market_by_condition_id(normalized)
            else:
                info = await polymarket_client.get_market_by_token_id(normalized)
        except Exception:
            info = None
        market_info_cache[normalized] = info
        return info

    for row in rows:
        market_id = _normalize_market_id(row.get("market_id"))
        if not market_id:
            row["outcome_labels"] = ["Yes", "No"]
            row["yes_label"] = "Yes"
            row["no_label"] = "No"
            continue

        info = await _load_market_info(market_id)
        labels = _extract_outcome_labels_from_market_info(
            info,
            fallback_slug=str(row.get("market_slug") or ""),
            fallback_question=str(row.get("market_question") or ""),
        )
        info_yes, info_no = _extract_outcome_prices_from_market_info(info)
        token_ids = (
            [str(token_id).strip().lower() for token_id in (info.get("token_ids") or []) if str(token_id or "").strip()]
            if isinstance(info, dict)
            else []
        )

        existing_yes = _safe_float(row.get("yes_price"))
        existing_no = _safe_float(row.get("no_price"))
        current_yes = existing_yes if existing_yes is not None else info_yes
        current_no = existing_no if existing_no is not None else info_no

        if current_yes is not None:
            row["yes_price"] = current_yes
            row["current_yes_price"] = current_yes
        elif row.get("current_yes_price") is None:
            row["current_yes_price"] = None

        if current_no is not None:
            row["no_price"] = current_no
            row["current_no_price"] = current_no
        elif row.get("current_no_price") is None:
            row["current_no_price"] = None

        row["outcome_labels"] = labels
        row["yes_label"] = labels[0]
        row["no_label"] = labels[1]
        row["market_token_ids"] = token_ids

    return rows


def _history_candidates_for_row(row: dict[str, Any]) -> list[str]:
    candidates: list[str] = []
    primary = _normalize_market_id(row.get("market_id"))
    if primary:
        candidates.extend([primary, primary.upper()])
    token_ids = row.get("market_token_ids")
    if isinstance(token_ids, list):
        for token_id in token_ids:
            token_norm = _normalize_market_id(token_id)
            if token_norm:
                candidates.extend([token_norm, token_norm.upper()])
    return list(dict.fromkeys(candidates))


def _attach_history_from_aliases(
    rows: list[dict[str, Any]],
    market_history: dict[str, list[dict]],
) -> None:
    if not rows or not market_history:
        return

    for row in rows:
        existing = row.get("price_history")
        if isinstance(existing, list) and len(existing) >= 2:
            continue

        for candidate in _history_candidates_for_row(row):
            history = market_history.get(candidate) or market_history.get(candidate.lower())
            if isinstance(history, list) and len(history) >= 2:
                compact = history[-20:]
                row["price_history"] = compact
                yes_price, no_price = _extract_yes_no_from_history(compact)
                if yes_price is not None:
                    row["yes_price"] = yes_price
                    row["current_yes_price"] = yes_price
                if no_price is not None:
                    row["no_price"] = no_price
                    row["current_no_price"] = no_price
                break


def _infer_yes_no_from_trade(
    *,
    outcome_hint: str,
    direction_hint: str,
    side: str,
    price: float,
) -> tuple[float, float]:
    side_u = side.upper()
    outcome_u = outcome_hint.upper()
    if not outcome_u:
        if direction_hint == "buy_yes":
            outcome_u = "YES"
        elif direction_hint == "buy_no":
            outcome_u = "NO"

    if outcome_u == "YES":
        if side_u == "SELL":
            return 1.0 - price, price
        return price, 1.0 - price

    if outcome_u == "NO":
        if side_u == "SELL":
            return price, 1.0 - price
        return 1.0 - price, price

    if side_u == "SELL":
        return 1.0 - price, price
    return price, 1.0 - price


async def _attach_activity_history_fallback(
    session: AsyncSession,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        return

    aliases_by_row: dict[int, list[str]] = {}
    requested_ids: set[str] = set()
    for idx, row in enumerate(rows):
        existing = row.get("price_history")
        if isinstance(existing, list) and len(existing) >= 2:
            continue
        aliases = _history_candidates_for_row(row)
        if not aliases:
            continue
        aliases_by_row[idx] = aliases
        requested_ids.update(alias.lower() for alias in aliases)

    if not requested_ids:
        return

    result = await session.execute(
        select(
            WalletActivityRollup.market_id,
            WalletActivityRollup.traded_at,
            WalletActivityRollup.side,
            WalletActivityRollup.price,
        )
        .where(func.lower(WalletActivityRollup.market_id).in_(list(requested_ids)))
        .order_by(WalletActivityRollup.traded_at.asc())
    )
    raw_rows = result.all()
    if not raw_rows:
        return

    events_by_market: dict[str, list[tuple[datetime, str, float]]] = defaultdict(list)
    for market_id, traded_at, side, price in raw_rows:
        market_norm = _normalize_market_id(market_id)
        if not market_norm or traded_at is None:
            continue
        price_f = _safe_float(price)
        if price_f is None or price_f < 0.0 or price_f > 1.01:
            continue
        events_by_market[market_norm].append((traded_at, str(side or ""), float(price_f)))

    for idx, aliases in aliases_by_row.items():
        row = rows[idx]
        outcome_hint = str(row.get("outcome") or "")
        direction_hint = str(row.get("direction") or "").strip().lower()
        points: list[dict[str, float]] = []
        for alias in aliases:
            market_events = events_by_market.get(alias.lower()) or []
            if market_events:
                for traded_at, side, price in market_events[-120:]:
                    yes_price, no_price = _infer_yes_no_from_trade(
                        outcome_hint=outcome_hint,
                        direction_hint=direction_hint,
                        side=side,
                        price=price,
                    )
                    points.append(
                        {
                            "t": float(traded_at.timestamp() * 1000.0),
                            "yes": round(yes_price, 6),
                            "no": round(no_price, 6),
                        }
                    )
                break

        if len(points) < 2:
            continue

        points.sort(key=lambda item: item["t"])
        compact = points[-20:]
        row["price_history"] = compact
        yes_last, no_last = _extract_yes_no_from_history(compact)
        if yes_last is not None:
            row["yes_price"] = yes_last
            row["current_yes_price"] = yes_last
        if no_last is not None:
            row["no_price"] = no_last
            row["current_no_price"] = no_last


async def _annotate_trader_signal_rows(
    session: AsyncSession,
    rows: list[dict],
) -> list[dict]:
    if not rows:
        return rows

    wallets_by_signal: list[list[str]] = []
    unique_addresses: set[str] = set()
    for row in rows:
        addresses = _extract_signal_wallet_addresses(row)
        wallets_by_signal.append(addresses)
        unique_addresses.update(addresses)

    pool_addresses: set[str] = set()
    tracked_addresses: set[str] = set()
    group_ids_by_address: dict[str, set[str]] = defaultdict(set)

    if unique_addresses:
        discovered_rows = await session.execute(
            select(DiscoveredWallet.address, DiscoveredWallet.in_top_pool).where(
                DiscoveredWallet.address.in_(list(unique_addresses))
            )
        )
        for address, in_top_pool in discovered_rows.all():
            normalized = _normalize_signal_wallet_address(address)
            if normalized and bool(in_top_pool):
                pool_addresses.add(normalized)

        tracked_rows = await session.execute(
            select(TrackedWallet.address).where(TrackedWallet.address.in_(list(unique_addresses)))
        )
        for (address,) in tracked_rows.all():
            normalized = _normalize_signal_wallet_address(address)
            if normalized:
                tracked_addresses.add(normalized)

        group_rows = await session.execute(
            select(TraderGroupMember.wallet_address, TraderGroupMember.group_id)
            .join(TraderGroup, TraderGroupMember.group_id == TraderGroup.id)
            .where(
                TraderGroup.is_active == True,  # noqa: E712
                TraderGroupMember.wallet_address.in_(list(unique_addresses)),
            )
        )
        for address, group_id in group_rows.all():
            normalized = _normalize_signal_wallet_address(address)
            if normalized and group_id:
                group_ids_by_address[normalized].add(str(group_id))

    for row, wallet_addresses in zip(rows, wallets_by_signal):
        pool_wallets = sum(1 for addr in wallet_addresses if addr in pool_addresses)
        tracked_wallets = sum(1 for addr in wallet_addresses if addr in tracked_addresses)
        group_wallets = sum(1 for addr in wallet_addresses if group_ids_by_address.get(addr))
        matched_group_ids = sorted(
            {group_id for addr in wallet_addresses for group_id in group_ids_by_address.get(addr, set())}
        )

        has_qualified_source = bool(pool_wallets or tracked_wallets or group_wallets)
        source_flags = {
            "from_pool": pool_wallets > 0,
            "from_tracked_traders": tracked_wallets > 0,
            "from_trader_groups": group_wallets > 0,
            "qualified": has_qualified_source,
        }
        source_breakdown = {
            "wallets_considered": len(wallet_addresses),
            "pool_wallets": pool_wallets,
            "tracked_wallets": tracked_wallets,
            "group_wallets": group_wallets,
            "group_count": len(matched_group_ids),
            "group_ids": matched_group_ids,
        }
        validation = _build_signal_validation_payload(
            signal=row,
            wallet_addresses=wallet_addresses,
            has_qualified_source=has_qualified_source,
        )

        row["source_flags"] = source_flags
        row["source_breakdown"] = source_breakdown
        row["validation"] = validation
        row["is_valid"] = bool(validation.get("is_valid"))
        row["is_actionable"] = bool(validation.get("is_actionable"))
        row["is_tradeable"] = bool(validation.get("is_tradeable"))
        row["validation_reasons"] = list(validation.get("reasons") or [])

    return rows


def _first_valid_trade_time(trade: dict) -> Optional[datetime]:
    for field in ("timestamp_iso", "match_time", "timestamp", "time", "created_at"):
        raw = trade.get(field)
        if raw is None:
            continue
        try:
            if isinstance(raw, (int, float)):
                return datetime.fromtimestamp(raw)
            text = str(raw).strip()
            if not text:
                continue
            if "T" in text or "-" in text:
                return datetime.fromisoformat(text.replace("Z", "+00:00").replace("+00:00", ""))
            return datetime.fromtimestamp(float(text))
        except Exception:
            continue
    return None


async def _track_wallet_addresses(
    addresses: list[str],
    label: str,
    fetch_initial: bool = False,
) -> int:
    tracked = 0
    for address in addresses:
        try:
            await wallet_tracker.add_wallet(
                address=address,
                label=label,
                fetch_initial=fetch_initial,
            )
            tracked += 1
        except Exception:
            continue
    return tracked


def _apply_pool_flag_updates(
    source_flags: dict,
    *,
    manual_include: Optional[bool] = None,
    manual_exclude: Optional[bool] = None,
    blacklisted: Optional[bool] = None,
    tracked_wallet: Optional[bool] = None,
) -> dict:
    out = dict(source_flags)
    if manual_include is not None:
        out[POOL_FLAG_MANUAL_INCLUDE] = bool(manual_include)
    if manual_exclude is not None:
        out[POOL_FLAG_MANUAL_EXCLUDE] = bool(manual_exclude)
    if blacklisted is not None:
        out[POOL_FLAG_BLACKLISTED] = bool(blacklisted)
    if tracked_wallet is not None:
        out["tracked_wallet"] = bool(tracked_wallet)
    return out


async def _fetch_group_payload(
    session: AsyncSession,
    include_members: bool = False,
    member_limit: int = 25,
) -> list[dict]:
    group_result = await session.execute(
        select(TraderGroup)
        .where(TraderGroup.is_active == True)  # noqa: E712
        .order_by(TraderGroup.created_at.asc())
    )
    groups = list(group_result.scalars().all())
    if not groups:
        return []

    group_ids = [g.id for g in groups]
    member_result = await session.execute(
        select(TraderGroupMember)
        .where(TraderGroupMember.group_id.in_(group_ids))
        .order_by(TraderGroupMember.added_at.asc())
    )
    members = list(member_result.scalars().all())

    members_by_group: dict[str, list[TraderGroupMember]] = defaultdict(list)
    all_addresses: set[str] = set()
    for member in members:
        members_by_group[member.group_id].append(member)
        all_addresses.add(member.wallet_address.lower())

    profile_map: dict[str, DiscoveredWallet] = {}
    if all_addresses:
        profile_rows = await session.execute(
            select(DiscoveredWallet).where(DiscoveredWallet.address.in_(list(all_addresses)))
        )
        for wallet in profile_rows.scalars().all():
            profile_map[wallet.address.lower()] = wallet

    payload: list[dict] = []
    for group in groups:
        group_members = members_by_group.get(group.id, [])
        item = {
            "id": group.id,
            "name": group.name,
            "description": group.description,
            "source_type": group.source_type,
            "suggestion_key": group.suggestion_key,
            "criteria": group.criteria or {},
            "auto_track_members": bool(group.auto_track_members),
            "member_count": len(group_members),
            "created_at": group.created_at.isoformat() if group.created_at else None,
            "updated_at": group.updated_at.isoformat() if group.updated_at else None,
        }
        if include_members:
            member_payload = []
            for member in group_members[:member_limit]:
                profile = profile_map.get(member.wallet_address.lower())
                member_payload.append(
                    {
                        "id": member.id,
                        "wallet_address": member.wallet_address,
                        "source": member.source,
                        "confidence": member.confidence,
                        "notes": member.notes,
                        "added_at": member.added_at.isoformat() if member.added_at else None,
                        "username": profile.username if profile else None,
                        "composite_score": (profile.composite_score if profile else None),
                        "quality_score": profile.quality_score if profile else None,
                        "activity_score": profile.activity_score if profile else None,
                        "pool_tier": profile.pool_tier if profile else None,
                    }
                )
            item["members"] = member_payload
        payload.append(item)

    return payload


# ==================== LEADERBOARD ====================


@discovery_router.get("/leaderboard")
async def get_leaderboard(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    min_trades: int = Query(default=0, ge=0),
    min_pnl: float = Query(default=0.0),
    insider_only: bool = Query(default=False),
    min_insider_score: Optional[float] = Query(default=None, ge=0.0, le=1.0),
    sort_by: str = Query(
        default="rank_score",
        description="rank_score, total_pnl, win_rate, sharpe_ratio, profit_factor, insider_score",
    ),
    sort_dir: str = Query(default="desc", description="asc or desc"),
    tags: Optional[str] = Query(default=None, description="Comma-separated tag filter"),
    recommendation: Optional[str] = Query(default=None, description="copy_candidate, monitor, avoid"),
    time_period: Optional[str] = Query(
        default=None,
        description="Time period filter: 24h, 7d, 30d, 90d, or all (default all)",
    ),
    active_within_hours: Optional[int] = Query(
        default=None,
        ge=1,
        le=720,
        description="Only include wallets with activity within last N hours",
    ),
    min_activity_score: Optional[float] = Query(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum activity score filter",
    ),
    pool_only: bool = Query(
        default=False,
        description="Only include wallets currently in the smart top pool",
    ),
    tier: Optional[str] = Query(
        default=None,
        description="Pool tier filter: core, rising, standby",
    ),
    search: Optional[str] = Query(
        default=None,
        description="Search wallet address, username, tags, or strategy terms",
    ),
    unique_entities_only: bool = Query(
        default=False,
        description="Deduplicate clustered wallets so each entity appears once",
    ),
    market_category: Optional[str] = Query(
        default=None,
        description="Market focus filter: politics, sports, crypto, culture, economics, tech, finance, weather",
    ),
):
    """
    Get the wallet leaderboard with comprehensive filters and sorting.

    Returns ranked wallets with trading stats, tags, and recommendations.
    Supports pagination, multi-field sorting, and tag-based filtering.
    """
    try:
        # Validate sort_by
        valid_sort_fields = [
            "rank_score",
            "composite_score",
            "quality_score",
            "activity_score",
            "stability_score",
            "last_trade_at",
            "total_pnl",
            "total_returned",
            "win_rate",
            "sharpe_ratio",
            "profit_factor",
            "total_trades",
            "avg_roi",
            "sortino_ratio",
            "trades_per_day",
            "insider_score",
        ]
        if sort_by not in valid_sort_fields:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid sort_by. Must be one of: {valid_sort_fields}",
            )

        # Validate sort_dir
        if sort_dir not in ("asc", "desc"):
            raise HTTPException(
                status_code=400,
                detail="Invalid sort_dir. Must be 'asc' or 'desc'",
            )

        # Validate recommendation
        valid_recommendations = ["copy_candidate", "monitor", "avoid"]
        if recommendation and recommendation not in valid_recommendations:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid recommendation. Must be one of: {valid_recommendations}",
            )

        # Validate market category (optional)
        valid_market_categories = [
            "politics",
            "sports",
            "crypto",
            "culture",
            "economics",
            "tech",
            "finance",
            "weather",
            "overall",
            "all",
        ]
        if market_category and market_category.lower() not in valid_market_categories:
            raise HTTPException(
                status_code=400,
                detail=(f"Invalid market_category. Must be one of: {valid_market_categories}"),
            )

        # Map time_period to rolling window key
        window_key = None
        if time_period and time_period != "all":
            window_key = TIME_PERIOD_TO_WINDOW_KEY.get(time_period)
            if window_key is None:
                valid_periods = list(TIME_PERIOD_TO_WINDOW_KEY.keys()) + ["all"]
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid time_period. Must be one of: {valid_periods}",
                )

        # Parse comma-separated tags
        tag_list = tags.split(",") if tags else None

        result = await wallet_discovery.get_leaderboard(
            limit=limit,
            offset=offset,
            min_trades=min_trades,
            min_pnl=min_pnl,
            insider_only=insider_only,
            min_insider_score=min_insider_score,
            sort_by=sort_by,
            sort_dir=sort_dir,
            tags=tag_list,
            recommendation=recommendation,
            window_key=window_key,
            active_within_hours=active_within_hours,
            min_activity_score=min_activity_score,
            pool_only=pool_only,
            tier=tier,
            search_text=search,
            unique_entities_only=unique_entities_only,
            market_category=market_category,
        )

        # When no discovered wallets yet, fall back to live Polymarket leaderboard so UI shows traders
        if (result.get("total") or 0) == 0 and offset == 0:
            try:
                from services.polymarket import polymarket_client

                raw = await polymarket_client.get_leaderboard(
                    limit=min(limit, 50),
                    time_period="ALL",
                    order_by="PNL",
                    category="OVERALL",
                )
                if raw:
                    wallets = [
                        {
                            "address": e.get("proxyWallet", ""),
                            "username": e.get("userName"),
                            "total_pnl": float(e.get("pnl") or 0),
                            "vol": float(e.get("vol") or 0),
                            "rank_position": int(e.get("rank", 0)) if str(e.get("rank", "")).isdigit() else None,
                            "tags": [],
                            "from_polymarket_live": True,
                        }
                        for e in raw
                        if e.get("proxyWallet")
                    ]
                    return {"wallets": wallets, "total": len(wallets), "from_polymarket_live": True}
            except Exception:
                pass

        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@discovery_router.get("/leaderboard/stats")
async def get_discovery_stats(session: AsyncSession = Depends(get_db_session)):
    """
    Get discovery engine statistics.

    Returns total wallets analyzed, last run timestamp, coverage metrics,
    and engine health information.
    """
    try:
        return await _build_discovery_status(session)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== WALLET PROFILES ====================


@discovery_router.get("/wallet/{wallet_address}/profile")
async def get_wallet_profile(wallet_address: str):
    """
    Get comprehensive wallet profile with all metrics, tags, cluster info, and rolling windows.

    Returns detailed analysis including trading statistics, detected strategies,
    tag classifications, entity cluster membership, and performance over
    multiple time windows.
    """
    try:
        address = validate_eth_address(wallet_address)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        profile = await wallet_discovery.get_wallet_profile(address)
        if not profile:
            raise HTTPException(status_code=404, detail="Wallet profile not found")
        return profile
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== DISCOVERY CONTROL ====================


@discovery_router.post("/run")
async def trigger_discovery(
    session: AsyncSession = Depends(get_db_session),
):
    """
    Queue a one-time discovery run for the discovery worker.
    """
    if global_pause_state.is_paused:
        raise HTTPException(
            status_code=409,
            detail="Global pause is active. Resume all workers before queueing runs.",
        )
    try:
        await discovery_shared_state.request_one_discovery_run(session)
        return {
            "status": "queued",
            "message": "Discovery run requested; worker will execute on next cycle.",
            **await discovery_shared_state.get_discovery_status_from_db(session),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@discovery_router.post("/start")
async def start_discovery_worker(session: AsyncSession = Depends(get_db_session)):
    """Resume automatic discovery worker cycles."""
    if global_pause_state.is_paused:
        raise HTTPException(
            status_code=409,
            detail="Global pause is active. Use /workers/resume-all first.",
        )
    await discovery_shared_state.set_discovery_paused(session, False)
    return {
        "status": "started",
        **await discovery_shared_state.get_discovery_status_from_db(session),
    }


@discovery_router.post("/pause")
async def pause_discovery_worker(session: AsyncSession = Depends(get_db_session)):
    """Pause automatic discovery worker cycles."""
    await discovery_shared_state.set_discovery_paused(session, True)
    return {
        "status": "paused",
        **await discovery_shared_state.get_discovery_status_from_db(session),
    }


@discovery_router.post("/interval")
async def set_discovery_interval(
    interval_minutes: int = Query(..., ge=5, le=1440),
    session: AsyncSession = Depends(get_db_session),
):
    """Set discovery worker cadence in minutes."""
    await discovery_shared_state.set_discovery_interval(session, interval_minutes)
    return {
        "status": "updated",
        **await discovery_shared_state.get_discovery_status_from_db(session),
    }


@discovery_router.post("/priority-backlog-mode")
async def set_discovery_priority_backlog_mode(
    enabled: bool = Query(..., description="Enable backlog-priority cadence and queue"),
    session: AsyncSession = Depends(get_db_session),
):
    """Enable or disable discovery priority backlog mode."""
    await discovery_shared_state.set_discovery_priority_backlog_mode(session, enabled)
    return {
        "status": "updated",
        **await discovery_shared_state.get_discovery_status_from_db(session),
    }


@discovery_router.post("/refresh-leaderboard")
async def trigger_refresh():
    """
    Force a leaderboard rank recalculation.

    Recomputes rank scores for all tracked wallets using the latest
    metrics without running a full discovery scan.
    """
    try:
        result = await wallet_discovery.refresh_leaderboard()
        return {
            "status": "success",
            "message": "Leaderboard refreshed",
            "result": result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== CONFLUENCE SIGNALS ====================


@discovery_router.get("/confluence")
async def get_confluence_signals(
    min_strength: float = Query(default=0.0, ge=0.0, le=1.0),
    min_tier: str = Query(default="WATCH", description="WATCH, HIGH, EXTREME"),
    limit: int = Query(default=50, ge=1, le=200),
):
    """
    Get active confluence signals.

    Identifies markets where multiple top-ranked wallets are converging
    on the same position. Higher strength indicates stronger agreement
    among skilled traders.
    """
    try:
        signals = await wallet_intelligence.confluence.get_active_signals(
            min_strength=min_strength,
            limit=limit,
            min_tier=min_tier,
        )
        return {"signals": signals}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@discovery_router.post("/confluence/scan")
async def trigger_confluence_scan():
    """
    Trigger a manual confluence scan.

    Analyzes current positions of top wallets to detect convergence
    patterns across active markets.
    """
    try:
        result = await wallet_intelligence.confluence.scan_for_confluence()
        return {
            "status": "success",
            "message": "Confluence scan completed",
            "result": result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@discovery_router.get("/pool/stats")
async def get_smart_pool_stats(session: AsyncSession = Depends(get_db_session)):
    """Get near-real-time smart wallet pool health metrics."""
    try:
        canonical = await smart_wallet_pool.get_pool_stats()
        worker = await read_worker_snapshot(session, "tracked_traders")
        stats = worker.get("stats") or {}
        if isinstance(stats, dict):
            pool_stats = stats.get("pool_stats")
            if isinstance(pool_stats, dict) and "active_1h" in pool_stats and "active_24h" in pool_stats:
                # Prefer DB-computed values for live counts, but preserve worker
                # timing metadata when canonical values are empty in API-only processes.
                merged = {**pool_stats, **canonical}
                for key in (
                    "last_pool_recompute_at",
                    "last_full_sweep_at",
                    "last_incremental_refresh_at",
                    "last_activity_reconciliation_at",
                    "last_error",
                    "candidates_last_sweep",
                    "events_last_reconcile",
                ):
                    if merged.get(key) in (None, "") and pool_stats.get(key) not in (None, ""):
                        merged[key] = pool_stats.get(key)
                return merged
        return canonical
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@discovery_router.post("/pool/actions/recompute")
async def recompute_smart_pool(
    mode: str = Query(
        default=POOL_RECOMPUTE_MODE_QUALITY_ONLY,
        description="quality_only or balanced",
    ),
):
    """Force smart pool recomputation with optional mode override."""
    normalized = str(mode or "").strip().lower()
    valid_modes = {POOL_RECOMPUTE_MODE_QUALITY_ONLY, POOL_RECOMPUTE_MODE_BALANCED}
    if normalized not in valid_modes:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode. Must be one of: {sorted(valid_modes)}",
        )
    try:
        await smart_wallet_pool.recompute_pool(mode=normalized)
        return {
            "status": "success",
            "mode": smart_wallet_pool.get_recompute_mode(),
            "pool_stats": await smart_wallet_pool.get_pool_stats(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@discovery_router.get("/pool/members")
async def get_pool_members(
    limit: int = Query(default=300, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    pool_only: bool = Query(default=True),
    include_blacklisted: bool = Query(default=True),
    tier: Optional[str] = Query(default=None, description="core or rising"),
    search: Optional[str] = Query(default=None),
    min_win_rate: float = Query(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Minimum win rate threshold (percent 0-100 or ratio 0-1).",
    ),
    sort_by: str = Query(
        default="composite_score",
        description=(
            "selection_score, composite_score, quality_score, activity_score, "
            "trades_24h, trades_1h, total_trades, total_pnl, win_rate, "
            "last_trade_at, rank_score"
        ),
    ),
    sort_dir: str = Query(default="desc", description="asc or desc"),
    session: AsyncSession = Depends(get_db_session),
):
    """List smart-pool members with actionable flags for management workflows."""
    try:
        valid_sort_fields = {
            "selection_score": lambda row: float(row.get("selection_score") or row.get("composite_score") or 0.0),
            "composite_score": lambda row: float(row.get("composite_score") or 0.0),
            "quality_score": lambda row: float(row.get("quality_score") or 0.0),
            "activity_score": lambda row: float(row.get("activity_score") or 0.0),
            "trades_24h": lambda row: int(row.get("trades_24h") or 0),
            "trades_1h": lambda row: int(row.get("trades_1h") or 0),
            "total_trades": lambda row: int(row.get("total_trades") or 0),
            "total_pnl": lambda row: float(row.get("total_pnl") or 0.0),
            "win_rate": lambda row: float(row.get("win_rate") or 0.0),
            "last_trade_at": lambda row: row.get("last_trade_at") or "",
            "rank_score": lambda row: float(row.get("rank_score") or 0.0),
        }
        if sort_by not in valid_sort_fields:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid sort_by. Must be one of: {list(valid_sort_fields)}",
            )
        if sort_dir not in {"asc", "desc"}:
            raise HTTPException(status_code=400, detail="sort_dir must be asc or desc")

        min_win_rate_ratio = _normalize_win_rate_ratio(min_win_rate)

        wallets_result = await session.execute(select(DiscoveredWallet))
        wallets = list(wallets_result.scalars().all())
        tracked_result = await session.execute(select(TrackedWallet.address, TrackedWallet.label))
        tracked_rows = list(tracked_result.all())
        tracked_addresses = {str(row[0]).lower() for row in tracked_rows}
        tracked_labels = {
            str(row[0]).lower(): str(row[1]).strip()
            for row in tracked_rows
            if row[1] is not None and str(row[1]).strip()
        }

        cluster_ids = sorted(
            {str(w.cluster_id).strip() for w in wallets if w.cluster_id is not None and str(w.cluster_id).strip()}
        )
        cluster_labels: dict[str, str] = {}
        if cluster_ids:
            cluster_rows = await session.execute(
                select(WalletCluster.id, WalletCluster.label).where(WalletCluster.id.in_(cluster_ids))
            )
            cluster_labels = {
                str(row[0]).strip().lower(): str(row[1]).strip()
                for row in cluster_rows.all()
                if row[1] is not None and str(row[1]).strip()
            }

        def _looks_placeholder_label(value: Optional[str]) -> bool:
            text = str(value or "").strip()
            if not text:
                return True
            lower = text.lower()
            if lower.startswith("0x") and ("..." in text or len(text) <= 16):
                return True
            return False

        query = (search or "").strip().lower()
        rows: list[dict] = []
        blacklisted_count = 0
        manual_included_count = 0
        manual_excluded_count = 0
        pool_member_count = 0
        tracked_in_pool_count = 0

        for wallet in wallets:
            source_flags = _coerce_source_flags(wallet.source_flags)
            flags = _pool_flags(source_flags)
            categories = _market_categories_from_flags(source_flags)
            addr_l = wallet.address.lower()
            is_tracked = addr_l in tracked_addresses
            tracked_label = tracked_labels.get(addr_l)
            cluster_label = cluster_labels.get(str(wallet.cluster_id or "").strip().lower())

            username = str(wallet.username or "").strip() or None
            display_name = username
            name_source = "username" if display_name else "unresolved"
            if not display_name and tracked_label and not _looks_placeholder_label(tracked_label):
                display_name = tracked_label
                name_source = "tracked_label"
            if not display_name and cluster_label:
                display_name = cluster_label
                name_source = "cluster_label"

            selection_meta = source_flags.get("pool_selection_meta")
            if not isinstance(selection_meta, dict):
                selection_meta = {}
            raw_reasons = selection_meta.get("reasons")
            selection_reasons = [r for r in raw_reasons if isinstance(r, dict)] if isinstance(raw_reasons, list) else []
            selection_breakdown = selection_meta.get("score_breakdown")
            if not isinstance(selection_breakdown, dict):
                selection_breakdown = {}
            try:
                selection_score = float(
                    selection_meta.get("selection_score")
                    if selection_meta.get("selection_score") is not None
                    else (wallet.composite_score or 0.0)
                )
            except Exception:
                selection_score = float(wallet.composite_score or 0.0)
            try:
                selection_rank = (
                    int(selection_meta.get("selection_rank"))
                    if selection_meta.get("selection_rank") is not None
                    else None
                )
            except Exception:
                selection_rank = None
            try:
                selection_percentile = (
                    float(selection_meta.get("selection_percentile"))
                    if selection_meta.get("selection_percentile") is not None
                    else None
                )
            except Exception:
                selection_percentile = None
            raw_eligibility_status = str(selection_meta.get("eligibility_status") or "").strip().lower()
            if raw_eligibility_status in {"eligible", "blocked"}:
                eligibility_status = raw_eligibility_status
            else:
                eligibility_status = "eligible" if wallet.in_top_pool else "blocked"
            raw_blockers = selection_meta.get("eligibility_blockers")
            if isinstance(raw_blockers, list):
                eligibility_blockers = [item for item in raw_blockers if isinstance(item, (dict, str))]
            else:
                eligibility_blockers = []
            try:
                analysis_freshness_hours = (
                    float(selection_meta.get("analysis_freshness_hours"))
                    if selection_meta.get("analysis_freshness_hours") is not None
                    else None
                )
            except Exception:
                analysis_freshness_hours = None
            if analysis_freshness_hours is None and wallet.last_analyzed_at is not None:
                analysis_freshness_hours = round(
                    max(
                        (utcnow() - wallet.last_analyzed_at).total_seconds(),
                        0.0,
                    )
                    / 3600.0,
                    2,
                )
            quality_gate_version = (
                str(selection_meta.get("quality_gate_version")).strip()
                if selection_meta.get("quality_gate_version") is not None
                else None
            )

            if flags["blacklisted"]:
                blacklisted_count += 1
            if flags["manual_include"]:
                manual_included_count += 1
            if flags["manual_exclude"]:
                manual_excluded_count += 1

            if pool_only and not wallet.in_top_pool:
                continue
            if not include_blacklisted and flags["blacklisted"]:
                continue
            if tier and (wallet.pool_tier or "").lower() != tier.lower():
                continue
            normalized_win_rate = _normalize_win_rate_ratio(wallet.win_rate)
            if normalized_win_rate < min_win_rate_ratio:
                continue

            tag_text = " ".join(wallet.tags or [])
            strategy_text = " ".join(wallet.strategies_detected or [])
            category_text = " ".join(categories)
            if query:
                haystack = " ".join(
                    [
                        wallet.address.lower(),
                        str(wallet.username or "").lower(),
                        str(display_name or "").lower(),
                        str(tracked_label or "").lower(),
                        str(wallet.pool_tier or "").lower(),
                        tag_text.lower(),
                        strategy_text.lower(),
                        category_text.lower(),
                    ]
                )
                if query not in haystack:
                    continue

            row = {
                "address": wallet.address,
                "username": wallet.username,
                "display_name": display_name,
                "name_source": name_source,
                "tracked_label": tracked_label,
                "cluster_label": cluster_label,
                "in_top_pool": bool(wallet.in_top_pool),
                "pool_tier": wallet.pool_tier,
                "pool_membership_reason": wallet.pool_membership_reason,
                "rank_score": wallet.rank_score or 0.0,
                "composite_score": wallet.composite_score or 0.0,
                "quality_score": wallet.quality_score or 0.0,
                "activity_score": wallet.activity_score or 0.0,
                "stability_score": wallet.stability_score or 0.0,
                "selection_score": selection_score,
                "selection_rank": selection_rank,
                "selection_percentile": selection_percentile,
                "selection_reasons": selection_reasons,
                "selection_breakdown": selection_breakdown,
                "selection_updated_at": selection_meta.get("updated_at"),
                "eligibility_status": eligibility_status,
                "eligibility_blockers": eligibility_blockers,
                "analysis_freshness_hours": analysis_freshness_hours,
                "quality_gate_version": quality_gate_version,
                "trades_1h": wallet.trades_1h or 0,
                "trades_24h": wallet.trades_24h or 0,
                "last_trade_at": wallet.last_trade_at.isoformat() if wallet.last_trade_at else None,
                "total_trades": wallet.total_trades or 0,
                "total_pnl": wallet.total_pnl or 0.0,
                "win_rate": normalized_win_rate,
                "tags": wallet.tags or [],
                "strategies_detected": wallet.strategies_detected or [],
                "market_categories": categories,
                "tracked_wallet": is_tracked,
                "pool_flags": flags,
            }
            rows.append(row)
            if row["in_top_pool"]:
                pool_member_count += 1
                if is_tracked:
                    tracked_in_pool_count += 1

        rows.sort(
            key=valid_sort_fields[sort_by],
            reverse=(sort_dir != "asc"),
        )

        total = len(rows)
        page = rows[offset : offset + limit]
        return {
            "total": total,
            "offset": offset,
            "limit": limit,
            "members": page,
            "stats": {
                "pool_members": pool_member_count,
                "blacklisted": blacklisted_count,
                "manual_included": manual_included_count,
                "manual_excluded": manual_excluded_count,
                "tracked_in_pool": tracked_in_pool_count,
                "tracked_total": len(tracked_addresses),
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@discovery_router.post("/pool/members/{wallet_address}/manual-include")
async def pool_manual_include(
    wallet_address: str,
    payload: Optional[PoolFlagRequest] = None,
    session: AsyncSession = Depends(get_db_session),
):
    """Force-include a wallet in the pool unless blacklisted."""
    try:
        address = validate_eth_address(wallet_address).lower()
        wallet = await session.get(DiscoveredWallet, address)
        if wallet is None:
            wallet = DiscoveredWallet(
                address=address,
                discovered_at=utcnow(),
                discovery_source="manual_pool",
            )
            session.add(wallet)

        source_flags = _coerce_source_flags(wallet.source_flags)
        source_flags = _apply_pool_flag_updates(
            source_flags,
            manual_include=True,
            manual_exclude=False,
            blacklisted=False,
        )
        if payload and payload.reason:
            source_flags["pool_manual_reason"] = payload.reason
        wallet.source_flags = source_flags
        await session.commit()

        await smart_wallet_pool.recompute_pool()
        return {"status": "success", "address": address, "pool_flags": _pool_flags(source_flags)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@discovery_router.delete("/pool/members/{wallet_address}/manual-include")
async def clear_pool_manual_include(
    wallet_address: str,
    session: AsyncSession = Depends(get_db_session),
):
    """Clear manual include override for a wallet."""
    try:
        address = validate_eth_address(wallet_address).lower()
        wallet = await session.get(DiscoveredWallet, address)
        if wallet is None:
            raise HTTPException(status_code=404, detail="Wallet not found")
        source_flags = _coerce_source_flags(wallet.source_flags)
        source_flags = _apply_pool_flag_updates(source_flags, manual_include=False)
        wallet.source_flags = source_flags
        await session.commit()

        await smart_wallet_pool.recompute_pool()
        return {"status": "success", "address": address, "pool_flags": _pool_flags(source_flags)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@discovery_router.post("/pool/members/{wallet_address}/manual-exclude")
async def pool_manual_exclude(
    wallet_address: str,
    payload: Optional[PoolFlagRequest] = None,
    session: AsyncSession = Depends(get_db_session),
):
    """Force-exclude a wallet from the pool."""
    try:
        address = validate_eth_address(wallet_address).lower()
        wallet = await session.get(DiscoveredWallet, address)
        if wallet is None:
            raise HTTPException(status_code=404, detail="Wallet not found")

        source_flags = _coerce_source_flags(wallet.source_flags)
        source_flags = _apply_pool_flag_updates(
            source_flags,
            manual_include=False,
            manual_exclude=True,
        )
        if payload and payload.reason:
            source_flags["pool_manual_reason"] = payload.reason
        wallet.source_flags = source_flags
        await session.commit()

        await smart_wallet_pool.recompute_pool()
        return {"status": "success", "address": address, "pool_flags": _pool_flags(source_flags)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@discovery_router.delete("/pool/members/{wallet_address}/manual-exclude")
async def clear_pool_manual_exclude(
    wallet_address: str,
    session: AsyncSession = Depends(get_db_session),
):
    """Clear manual exclude override for a wallet."""
    try:
        address = validate_eth_address(wallet_address).lower()
        wallet = await session.get(DiscoveredWallet, address)
        if wallet is None:
            raise HTTPException(status_code=404, detail="Wallet not found")
        source_flags = _coerce_source_flags(wallet.source_flags)
        source_flags = _apply_pool_flag_updates(source_flags, manual_exclude=False)
        wallet.source_flags = source_flags
        await session.commit()

        await smart_wallet_pool.recompute_pool()
        return {"status": "success", "address": address, "pool_flags": _pool_flags(source_flags)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@discovery_router.post("/pool/members/{wallet_address}/blacklist")
async def blacklist_pool_wallet(
    wallet_address: str,
    payload: Optional[PoolFlagRequest] = None,
    session: AsyncSession = Depends(get_db_session),
):
    """Blacklist wallet from pool selection."""
    try:
        address = validate_eth_address(wallet_address).lower()
        wallet = await session.get(DiscoveredWallet, address)
        if wallet is None:
            raise HTTPException(status_code=404, detail="Wallet not found")
        source_flags = _coerce_source_flags(wallet.source_flags)
        source_flags = _apply_pool_flag_updates(
            source_flags,
            blacklisted=True,
            manual_include=False,
            manual_exclude=True,
        )
        if payload and payload.reason:
            source_flags["pool_blacklist_reason"] = payload.reason
        wallet.source_flags = source_flags
        await session.commit()

        await smart_wallet_pool.recompute_pool()
        return {"status": "success", "address": address, "pool_flags": _pool_flags(source_flags)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@discovery_router.delete("/pool/members/{wallet_address}/blacklist")
async def unblacklist_pool_wallet(
    wallet_address: str,
    session: AsyncSession = Depends(get_db_session),
):
    """Remove wallet from pool blacklist."""
    try:
        address = validate_eth_address(wallet_address).lower()
        wallet = await session.get(DiscoveredWallet, address)
        if wallet is None:
            raise HTTPException(status_code=404, detail="Wallet not found")
        source_flags = _coerce_source_flags(wallet.source_flags)
        source_flags = _apply_pool_flag_updates(source_flags, blacklisted=False)
        wallet.source_flags = source_flags
        await session.commit()

        await smart_wallet_pool.recompute_pool()
        return {"status": "success", "address": address, "pool_flags": _pool_flags(source_flags)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@discovery_router.delete("/pool/members/{wallet_address}")
async def delete_pool_wallet(
    wallet_address: str,
    session: AsyncSession = Depends(get_db_session),
):
    """Delete wallet from discovery/tracking datasets and remove from pool."""
    try:
        address = validate_eth_address(wallet_address).lower()

        removed_discovered = (
            await session.execute(delete(DiscoveredWallet).where(DiscoveredWallet.address == address))
        ).rowcount or 0
        removed_rollups = (
            await session.execute(delete(WalletActivityRollup).where(WalletActivityRollup.wallet_address == address))
        ).rowcount or 0
        removed_tracked = (
            await session.execute(delete(TrackedWallet).where(TrackedWallet.address == address))
        ).rowcount or 0
        removed_group_memberships = (
            await session.execute(delete(TraderGroupMember).where(TraderGroupMember.wallet_address == address))
        ).rowcount or 0
        await session.commit()

        await smart_wallet_pool.recompute_pool()
        return {
            "status": "success",
            "address": address,
            "removed": {
                "discovered_wallets": int(removed_discovered),
                "wallet_activity_rollups": int(removed_rollups),
                "tracked_wallets": int(removed_tracked),
                "group_memberships": int(removed_group_memberships),
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@discovery_router.post("/pool/actions/promote-tracked")
async def promote_tracked_wallets_to_pool(
    limit: int = Query(default=300, ge=1, le=2000),
    session: AsyncSession = Depends(get_db_session),
):
    """Promote tracked-wallet list into manual-included pool candidates."""
    try:
        tracked_rows = (
            (await session.execute(select(TrackedWallet).order_by(TrackedWallet.added_at.asc()).limit(limit)))
            .scalars()
            .all()
        )
        tracked_addresses = [row.address.lower() for row in tracked_rows if row.address]
        if not tracked_addresses:
            return {"status": "success", "promoted": 0, "created": 0, "updated": 0}

        existing_rows = (
            (await session.execute(select(DiscoveredWallet).where(DiscoveredWallet.address.in_(tracked_addresses))))
            .scalars()
            .all()
        )
        existing = {row.address.lower(): row for row in existing_rows}

        created = 0
        updated = 0
        for address in tracked_addresses:
            wallet = existing.get(address)
            if wallet is None:
                wallet = DiscoveredWallet(
                    address=address,
                    discovered_at=utcnow(),
                    discovery_source="manual_pool",
                )
                session.add(wallet)
                created += 1
            else:
                updated += 1

            source_flags = _coerce_source_flags(wallet.source_flags)
            source_flags = _apply_pool_flag_updates(
                source_flags,
                manual_include=True,
                manual_exclude=False,
                blacklisted=False,
                tracked_wallet=True,
            )
            wallet.source_flags = source_flags

        await session.commit()
        await smart_wallet_pool.recompute_pool()

        return {
            "status": "success",
            "promoted": len(tracked_addresses),
            "created": created,
            "updated": updated,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@discovery_router.get("/traders")
async def get_traders_overview(
    tracked_limit: int = Query(default=200, ge=1, le=500),
    confluence_limit: int = Query(default=50, ge=1, le=200),
    min_tier: str = Query(default="WATCH", description="WATCH, HIGH, EXTREME"),
    hours: int = Query(
        default=24,
        ge=1,
        le=168,
        description="Tracked trade activity window in hours",
    ),
    session: AsyncSession = Depends(get_db_session),
):
    """Broader traders view: tracked traders + discovery confluence + groups."""
    try:
        tracked_wallets = await wallet_tracker.get_all_wallets()
        cutoff = utcnow() - timedelta(hours=hours)

        tracked_rows: list[dict] = []
        total_recent_trades = 0
        for wallet in tracked_wallets:
            trades = wallet.get("recent_trades") or []
            latest_dt: Optional[datetime] = None
            recent_trades = 0
            for trade in trades:
                trade_dt = _first_valid_trade_time(trade)
                if trade_dt:
                    if latest_dt is None or trade_dt > latest_dt:
                        latest_dt = trade_dt
                    if trade_dt >= cutoff:
                        recent_trades += 1

            total_recent_trades += recent_trades
            tracked_rows.append(
                {
                    "address": wallet.get("address"),
                    "label": wallet.get("label"),
                    "username": wallet.get("username"),
                    "recent_trade_count": recent_trades,
                    "latest_trade_at": latest_dt.isoformat() if latest_dt else None,
                    "open_positions": len(wallet.get("positions") or []),
                }
            )

        tracked_rows.sort(
            key=lambda row: (
                row["recent_trade_count"],
                row.get("latest_trade_at") or "",
            ),
            reverse=True,
        )

        confluence_signals = await smart_wallet_pool.get_tracked_trader_opportunities(
            limit=confluence_limit,
            min_tier=min_tier,
        )
        market_history = await _load_scanner_market_history(session)
        _attach_market_history_to_signal_rows(confluence_signals, market_history)
        await _attach_signal_market_metadata(confluence_signals)
        _attach_history_from_aliases(confluence_signals, market_history)
        await _attach_activity_history_fallback(session, confluence_signals)
        await _annotate_trader_signal_rows(session, confluence_signals)

        groups = await _fetch_group_payload(session=session, include_members=False)

        return {
            "tracked": {
                "wallets": tracked_rows[:tracked_limit],
                "total_wallets": len(tracked_rows),
                "hours_window": hours,
                "recent_trade_count": total_recent_trades,
            },
            "groups": {
                "items": groups,
                "total_groups": len(groups),
                "total_members": sum(g.get("member_count", 0) for g in groups),
            },
            "confluence": {
                "signals": confluence_signals,
                "total_signals": len(confluence_signals),
                "min_tier": min_tier.upper(),
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@discovery_router.get("/opportunities/tracked-traders")
async def get_tracked_trader_opportunities(
    limit: int = Query(default=50, ge=1, le=200),
    min_tier: str = Query(default="WATCH", description="WATCH, HIGH, EXTREME"),
    session: AsyncSession = Depends(get_db_session),
):
    """Legacy alias for confluence signals powering the Traders surface."""
    try:
        opportunities = await smart_wallet_pool.get_tracked_trader_opportunities(
            limit=limit,
            min_tier=min_tier,
        )
        if not opportunities:
            # Self-heal path: run an on-demand confluence scan so the UI can
            # recover quickly after stale/expired signal windows.
            try:
                await wallet_intelligence.confluence.scan_for_confluence()
                opportunities = await smart_wallet_pool.get_tracked_trader_opportunities(
                    limit=limit,
                    min_tier=min_tier,
                )
            except Exception:
                pass
        market_history = await _load_scanner_market_history(session)
        _attach_market_history_to_signal_rows(opportunities, market_history)
        await _attach_signal_market_metadata(opportunities)
        _attach_history_from_aliases(opportunities, market_history)
        await _attach_activity_history_fallback(session, opportunities)
        await _annotate_trader_signal_rows(session, opportunities)
        return {"opportunities": opportunities, "total": len(opportunities)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@discovery_router.get("/opportunities/insider")
async def get_insider_opportunities(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    min_confidence: float = Query(default=0.0, ge=0.0, le=1.0),
    direction: Optional[str] = Query(default=None, description="buy_yes or buy_no"),
    max_age_minutes: int = Query(default=180, ge=1, le=1440),
    session: AsyncSession = Depends(get_db_session),
):
    """List pending/submitted insider opportunities built from flagged-wallet behavior."""
    if direction and direction not in {"buy_yes", "buy_no"}:
        raise HTTPException(status_code=400, detail="direction must be buy_yes or buy_no")
    try:
        payload = await insider_detector.list_opportunities(
            limit=limit,
            offset=offset,
            min_confidence=min_confidence,
            direction=direction,
            max_age_minutes=max_age_minutes,
        )
        market_history = await _load_scanner_market_history(session)
        opportunities = payload.get("opportunities") if isinstance(payload, dict) else None
        if isinstance(opportunities, list):
            _attach_market_history_to_signal_rows(opportunities, market_history)
            await _attach_signal_market_metadata(opportunities)
            _attach_history_from_aliases(opportunities, market_history)
            await _attach_activity_history_fallback(session, opportunities)
            await _annotate_trader_signal_rows(session, opportunities)
        return payload
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@discovery_router.get("/insider/intents")
async def get_insider_intents(
    status_filter: Optional[str] = Query(
        default=None,
        description="pending, submitted, executed, skipped, expired",
    ),
    limit: int = Query(default=100, ge=1, le=1000),
):
    """List insider trade intents for the trader orchestrator pipeline."""
    if status_filter and status_filter not in {
        "pending",
        "submitted",
        "executed",
        "skipped",
        "expired",
    }:
        raise HTTPException(
            status_code=400,
            detail="Invalid status_filter. Must be pending|submitted|executed|skipped|expired",
        )
    try:
        rows = await insider_detector.list_intents(status_filter=status_filter, limit=limit)
        intents = [
            {
                "id": row.id,
                "signal_key": row.signal_key,
                "market_id": row.market_id,
                "market_question": row.market_question,
                "direction": row.direction,
                "entry_price": row.entry_price,
                "edge_percent": row.edge_percent,
                "confidence": row.confidence,
                "insider_score": row.insider_score,
                "wallet_addresses": row.wallet_addresses_json or [],
                "suggested_size_usd": row.suggested_size_usd,
                "metadata": row.metadata_json or {},
                "status": row.status,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "consumed_at": row.consumed_at.isoformat() if row.consumed_at else None,
            }
            for row in rows
        ]
        return {"total": len(intents), "intents": intents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== TRADER GROUPS ====================


@discovery_router.get("/groups")
async def get_trader_groups(
    include_members: bool = Query(default=False),
    member_limit: int = Query(default=25, ge=1, le=200),
    session: AsyncSession = Depends(get_db_session),
):
    """List tracked trader groups."""
    try:
        groups = await _fetch_group_payload(
            session=session,
            include_members=include_members,
            member_limit=member_limit,
        )
        return {"groups": groups, "total": len(groups)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@discovery_router.post("/groups")
async def create_trader_group(
    payload: CreateTraderGroupRequest,
    session: AsyncSession = Depends(get_db_session),
):
    """Create a trader group and optionally add all members to tracking."""
    try:
        name = payload.name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="Group name is required")

        source_type = payload.source_type.strip().lower()
        if source_type not in GROUP_SOURCE_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid source_type. Must be one of: {sorted(GROUP_SOURCE_TYPES)}",
            )

        existing = await session.execute(select(TraderGroup).where(func.lower(TraderGroup.name) == name.lower()))
        if existing.scalars().first():
            raise HTTPException(
                status_code=409,
                detail=f"Trader group '{name}' already exists",
            )

        try:
            addresses = _normalize_wallet_addresses(payload.wallet_addresses)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        group = TraderGroup(
            id=str(uuid.uuid4()),
            name=name,
            description=(payload.description or None),
            source_type=source_type,
            suggestion_key=payload.suggestion_key,
            criteria=payload.criteria or {},
            auto_track_members=bool(payload.auto_track_members),
            is_active=True,
        )
        session.add(group)
        for address in addresses:
            session.add(
                TraderGroupMember(
                    id=str(uuid.uuid4()),
                    group_id=group.id,
                    wallet_address=address,
                    source=payload.source_label,
                )
            )
        await session.commit()

        tracked_count = 0
        if payload.auto_track_members and addresses:
            tracked_count = await _track_wallet_addresses(
                addresses=addresses,
                label=f"Group: {name}",
                fetch_initial=False,
            )

        groups = await _fetch_group_payload(
            session=session,
            include_members=True,
            member_limit=200,
        )
        created = next((g for g in groups if g.get("id") == group.id), None)

        return {
            "status": "success",
            "group": created,
            "tracked_members": tracked_count,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@discovery_router.post("/groups/{group_id}/members")
async def add_group_members(
    group_id: str,
    payload: UpdateTraderGroupMembersRequest,
    session: AsyncSession = Depends(get_db_session),
):
    """Add wallets to an existing trader group."""
    try:
        group = await session.get(TraderGroup, group_id)
        if not group or not group.is_active:
            raise HTTPException(status_code=404, detail="Trader group not found")

        try:
            addresses = _normalize_wallet_addresses(payload.wallet_addresses)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        if not addresses:
            return {"status": "success", "added_members": 0, "tracked_members": 0}

        existing_members = await session.execute(
            select(TraderGroupMember.wallet_address).where(TraderGroupMember.group_id == group_id)
        )
        existing_addresses = {
            str(address).lower() for address in existing_members.scalars().all() if isinstance(address, str)
        }

        to_add = [addr for addr in addresses if addr not in existing_addresses]
        for address in to_add:
            session.add(
                TraderGroupMember(
                    id=str(uuid.uuid4()),
                    group_id=group_id,
                    wallet_address=address,
                    source=payload.source_label,
                )
            )
        await session.commit()

        tracked_count = 0
        if payload.add_to_tracking and to_add:
            tracked_count = await _track_wallet_addresses(
                addresses=to_add,
                label=f"Group: {group.name}",
                fetch_initial=False,
            )

        groups = await _fetch_group_payload(
            session=session,
            include_members=True,
            member_limit=200,
        )
        updated = next((g for g in groups if g.get("id") == group_id), None)

        return {
            "status": "success",
            "added_members": len(to_add),
            "tracked_members": tracked_count,
            "group": updated,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@discovery_router.delete("/groups/{group_id}/members/{wallet_address}")
async def remove_group_member(
    group_id: str,
    wallet_address: str,
    session: AsyncSession = Depends(get_db_session),
):
    """Remove a wallet from a trader group."""
    try:
        try:
            address = validate_eth_address(wallet_address).lower()
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        member_result = await session.execute(
            select(TraderGroupMember).where(
                TraderGroupMember.group_id == group_id,
                TraderGroupMember.wallet_address == address,
            )
        )
        member = member_result.scalars().first()
        if not member:
            raise HTTPException(status_code=404, detail="Group member not found")

        await session.delete(member)
        await session.commit()
        return {
            "status": "success",
            "group_id": group_id,
            "wallet_address": address,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@discovery_router.delete("/groups/{group_id}")
async def delete_trader_group(
    group_id: str,
    session: AsyncSession = Depends(get_db_session),
):
    """Delete a trader group and all its memberships."""
    try:
        group = await session.get(TraderGroup, group_id)
        if not group:
            raise HTTPException(status_code=404, detail="Trader group not found")

        members_result = await session.execute(select(TraderGroupMember).where(TraderGroupMember.group_id == group_id))
        for member in members_result.scalars().all():
            await session.delete(member)
        await session.delete(group)
        await session.commit()
        return {"status": "success", "group_id": group_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@discovery_router.post("/groups/{group_id}/track")
async def track_group_members(
    group_id: str,
    session: AsyncSession = Depends(get_db_session),
):
    """Ensure all wallets in a trader group are tracked by the wallet monitor."""
    try:
        group = await session.get(TraderGroup, group_id)
        if not group or not group.is_active:
            raise HTTPException(status_code=404, detail="Trader group not found")

        member_result = await session.execute(
            select(TraderGroupMember.wallet_address).where(TraderGroupMember.group_id == group_id)
        )
        addresses = [addr for addr in member_result.scalars().all() if isinstance(addr, str)]

        tracked_count = await _track_wallet_addresses(
            addresses=addresses,
            label=f"Group: {group.name}",
            fetch_initial=False,
        )
        return {
            "status": "success",
            "group_id": group_id,
            "tracked_members": tracked_count,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@discovery_router.get("/groups/suggestions")
async def get_group_suggestions(
    min_group_size: int = Query(default=3, ge=2, le=100),
    max_suggestions: int = Query(default=12, ge=1, le=100),
    min_composite_score: float = Query(default=0.55, ge=0.0, le=1.0),
    session: AsyncSession = Depends(get_db_session),
):
    """Suggest high-quality trader groups from discovery clusters/tags/pool tiers."""
    try:
        suggestions: list[dict] = []

        existing_result = await session.execute(select(TraderGroup.name))
        existing_names = {
            str(name).strip().lower() for name in existing_result.scalars().all() if isinstance(name, str)
        }

        # 1) Cluster-driven suggestions (entity-linked wallets)
        cluster_result = await session.execute(
            select(WalletCluster)
            .where(WalletCluster.total_wallets >= min_group_size)
            .order_by(WalletCluster.combined_pnl.desc())
            .limit(max_suggestions * 2)
        )
        for cluster in cluster_result.scalars().all():
            wallet_rows = await session.execute(
                select(DiscoveredWallet)
                .where(
                    DiscoveredWallet.cluster_id == cluster.id,
                    DiscoveredWallet.composite_score >= min_composite_score,
                )
                .order_by(DiscoveredWallet.composite_score.desc())
                .limit(200)
            )
            wallets = list(wallet_rows.scalars().all())
            if len(wallets) < min_group_size:
                continue

            addresses = [w.address.lower() for w in wallets]
            name = (cluster.label or f"Cluster {cluster.id[:8]}").strip()
            avg_comp = sum((w.composite_score or 0.0) for w in wallets) / len(wallets)
            suggestions.append(
                {
                    "id": _suggestion_id("cluster", cluster.id),
                    "kind": "cluster",
                    "name": name,
                    "description": (
                        f"Entity-linked wallets discovered via clustering "
                        f"(confidence {round((cluster.confidence or 0.0) * 100)}%)."
                    ),
                    "wallet_count": len(addresses),
                    "wallet_addresses": addresses,
                    "avg_composite_score": round(avg_comp, 4),
                    "sample_wallets": [
                        {
                            "address": w.address,
                            "username": w.username,
                            "composite_score": w.composite_score or 0.0,
                            "pool_tier": w.pool_tier,
                        }
                        for w in wallets[:8]
                    ],
                    "criteria": {
                        "cluster_id": cluster.id,
                        "confidence": cluster.confidence or 0.0,
                    },
                    "already_exists": name.lower() in existing_names,
                }
            )

        # 2) Pool tier suggestions (core/rising/standby quality cohorts)
        for tier in ("core", "rising", "standby"):
            tier_rows = await session.execute(
                select(DiscoveredWallet)
                .where(
                    DiscoveredWallet.in_top_pool == True,  # noqa: E712
                    func.lower(DiscoveredWallet.pool_tier) == tier,
                    DiscoveredWallet.composite_score >= min_composite_score,
                )
                .order_by(
                    DiscoveredWallet.composite_score.desc(),
                    DiscoveredWallet.last_trade_at.desc(),
                )
                .limit(200)
            )
            wallets = list(tier_rows.scalars().all())
            if len(wallets) < min_group_size:
                continue

            name = f"{tier.title()} Pool Traders"
            addresses = [w.address.lower() for w in wallets]
            avg_comp = sum((w.composite_score or 0.0) for w in wallets) / len(wallets)
            suggestions.append(
                {
                    "id": _suggestion_id("pool", tier),
                    "kind": "pool_tier",
                    "name": name,
                    "description": (f"Auto-group from discovery smart pool tier '{tier}'."),
                    "wallet_count": len(addresses),
                    "wallet_addresses": addresses,
                    "avg_composite_score": round(avg_comp, 4),
                    "sample_wallets": [
                        {
                            "address": w.address,
                            "username": w.username,
                            "composite_score": w.composite_score or 0.0,
                            "pool_tier": w.pool_tier,
                        }
                        for w in wallets[:8]
                    ],
                    "criteria": {"pool_tier": tier},
                    "already_exists": name.lower() in existing_names,
                }
            )

        # 3) Tag-driven suggestions from high-quality discovered wallets
        tag_candidate_rows = await session.execute(
            select(DiscoveredWallet)
            .where(DiscoveredWallet.composite_score >= min_composite_score)
            .order_by(DiscoveredWallet.composite_score.desc())
            .limit(1500)
        )
        wallets = list(tag_candidate_rows.scalars().all())
        wallets_by_tag: dict[str, list[DiscoveredWallet]] = defaultdict(list)
        for wallet in wallets:
            for tag in wallet.tags or []:
                if isinstance(tag, str) and tag.strip():
                    wallets_by_tag[tag.strip().lower()].append(wallet)

        for tag, tagged_wallets in wallets_by_tag.items():
            unique_wallets = {w.address.lower(): w for w in tagged_wallets}
            rows = sorted(
                unique_wallets.values(),
                key=lambda w: w.composite_score or 0.0,
                reverse=True,
            )
            if len(rows) < min_group_size:
                continue

            display_tag = tag.replace("_", " ").title()
            name = f"{display_tag} Traders"
            addresses = [w.address.lower() for w in rows[:200]]
            avg_comp = sum((w.composite_score or 0.0) for w in rows[:200]) / len(addresses)
            suggestions.append(
                {
                    "id": _suggestion_id("tag", tag),
                    "kind": "tag",
                    "name": name,
                    "description": (f"High-quality discovered traders sharing tag '{tag}'."),
                    "wallet_count": len(addresses),
                    "wallet_addresses": addresses,
                    "avg_composite_score": round(avg_comp, 4),
                    "sample_wallets": [
                        {
                            "address": w.address,
                            "username": w.username,
                            "composite_score": w.composite_score or 0.0,
                            "pool_tier": w.pool_tier,
                        }
                        for w in rows[:8]
                    ],
                    "criteria": {"tag": tag},
                    "already_exists": name.lower() in existing_names,
                }
            )

        suggestions.sort(
            key=lambda s: (s["wallet_count"], s.get("avg_composite_score") or 0.0),
            reverse=True,
        )

        # De-duplicate by suggestion id and trim.
        deduped: dict[str, dict] = {}
        for suggestion in suggestions:
            sid = suggestion["id"]
            if sid in deduped:
                continue
            deduped[sid] = suggestion
            if len(deduped) >= max_suggestions:
                break

        return {"suggestions": list(deduped.values()), "total": len(deduped)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== ENTITY CLUSTERS ====================


@discovery_router.get("/clusters")
async def get_clusters(
    min_wallets: int = Query(default=2, ge=2, le=100),
):
    """
    Get wallet clusters (groups of wallets belonging to the same entity).

    Uses on-chain analysis to identify wallets that are likely controlled
    by the same person or organization based on funding patterns,
    coordinated trading, and timing analysis.
    """
    try:
        clusters = await wallet_intelligence.clusterer.get_clusters(
            min_wallets=min_wallets,
        )
        return {"clusters": clusters}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@discovery_router.get("/clusters/{cluster_id}")
async def get_cluster_detail(cluster_id: str):
    """
    Get detailed info about a specific cluster and its member wallets.

    Returns all wallets in the cluster, evidence linking them, shared
    trading patterns, and aggregate performance metrics.
    """
    try:
        detail = await wallet_intelligence.clusterer.get_cluster_detail(
            cluster_id=cluster_id,
        )
        if not detail:
            raise HTTPException(status_code=404, detail="Cluster not found")
        return detail
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== TAGS ====================


@discovery_router.get("/tags")
async def get_all_tags():
    """
    Get all tag definitions with wallet counts.

    Tags classify wallets by behavior (e.g., whale, sniper, market_maker,
    arbitrageur). Each tag includes a description and the number of
    wallets currently carrying that tag.
    """
    try:
        tags = await wallet_intelligence.tagger.get_all_tags()
        return {"tags": tags}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@discovery_router.get("/tags/{tag_name}/wallets")
async def get_wallets_by_tag(
    tag_name: str,
    limit: int = Query(default=100, ge=1, le=500),
):
    """
    Get wallets with a specific tag.

    Returns a list of wallets classified under the given tag,
    sorted by rank score.
    """
    try:
        wallets = await wallet_intelligence.tagger.get_wallets_by_tag(
            tag_name=tag_name,
            limit=limit,
        )
        return {"wallets": wallets}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== CROSS-PLATFORM ====================


@discovery_router.get("/cross-platform")
async def get_cross_platform_entities(
    limit: int = Query(default=50, ge=1, le=200),
):
    """
    Get entities tracked across Polymarket and Kalshi.

    Identifies traders operating on multiple prediction market platforms,
    enabling detection of cross-platform arbitrage strategies and
    providing a more complete view of trader behavior.
    """
    try:
        entities = await wallet_intelligence.cross_platform.get_cross_platform_entities(
            limit=limit,
        )
        return entities
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@discovery_router.get("/cross-platform/arb-activity")
async def get_cross_platform_arb():
    """
    Get recent cross-platform arbitrage activity.

    Returns instances where entities are exploiting price differences
    between Polymarket and Kalshi on the same underlying events.
    """
    try:
        activity = await wallet_intelligence.cross_platform.get_cross_platform_arb_activity()
        return activity
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
