from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.params import Param
from typing import Any, Literal, Optional
from collections.abc import Mapping
from datetime import datetime, timezone
from utils.utcnow import utcnow, utcfromtimestamp
from sqlalchemy.ext.asyncio import AsyncSession

from sqlalchemy import select
from models import Opportunity
from models.database import (
    AsyncSessionLocal,
    ScannerSnapshot,
    Strategy,
    StrategyValidationProfile,
    get_db_session,
    recover_pool,
)
from models.opportunity import OpportunityFilter
from services import polymarket_client
from services.wallet_tracker import wallet_tracker
from services.kalshi_client import kalshi_client
from services.opportunity_strategy_catalog import ensure_system_opportunity_strategies_seeded
from services import shared_state
from services.pause_state import global_pause_state
from utils.logger import get_logger
from utils.market_urls import attach_market_links_to_opportunity_dict, serialize_opportunity_with_links

router = APIRouter()
logger = get_logger("routes")

DEFAULT_STRATEGY_META = {
    "domain": "event_markets",
    "timeframe": "event",
    "sources": ["scanner"],
}

STRATEGY_META_BY_TYPE: dict[str, dict[str, object]] = {
    "btc_eth_maker_quote": {
        "domain": "crypto",
        "timeframe": "5m/15m",
        "sources": ["crypto"],
    },
    "btc_eth_directional_edge": {
        "domain": "crypto",
        "timeframe": "5m/15m",
        "sources": ["crypto"],
    },
    "btc_eth_convergence": {
        "domain": "crypto",
        "timeframe": "5m/15m",
        "sources": ["crypto"],
    },
    "news_edge": {
        "domain": "event_markets",
        "timeframe": "event",
        "sources": ["news"],
    },
    "weather_edge": {
        "domain": "event_markets",
        "timeframe": "forecast",
        "sources": ["weather"],
    },
    "event_driven": {
        "domain": "event_markets",
        "timeframe": "event",
        "sources": ["scanner", "events"],
    },
    "ctf_basic_arb": {
        "domain": "polymarket_ctf",
        "timeframe": "event",
        "sources": ["scanner", "events"],
    },
    "stat_arb": {
        "domain": "event_markets",
        "timeframe": "event",
        "sources": ["scanner", "events"],
    },
}

DB_RETRY_ATTEMPTS = 3
DB_RETRY_BASE_DELAY_SECONDS = 0.2
DB_RETRY_MAX_DELAY_SECONDS = 1.5
OPPORTUNITY_PAGE_PREFETCH_MULTIPLIER = 2


from utils.retry import is_retryable_db_error as _is_retryable_db_error  # noqa: E402
from utils.retry import db_retry_delay as _db_retry_delay  # noqa: E402


def _strategy_meta(strategy_type: str, *, is_plugin: bool) -> dict[str, object]:
    base = dict(DEFAULT_STRATEGY_META)
    base.update(STRATEGY_META_BY_TYPE.get(strategy_type, {}))
    if is_plugin:
        base["domain"] = "event_markets"
        base["timeframe"] = "plugin"
        base["sources"] = ["scanner"]
    return base


def _normalize_sub_strategy(value: Optional[str]) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    return text.replace("-", "_").replace(" ", "_")


async def _resolve_wallet_address(wallet_identifier: str) -> str:
    try:
        resolved = await polymarket_client.resolve_wallet_identifier(wallet_identifier)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return str(resolved.get("address") or "").lower()


def _derive_opportunity_sub_strategy(opportunity: object) -> Optional[str]:
    """Derive strategy-specific subtype from opportunity metadata/title."""

    def _field(name: str, default=None):
        if isinstance(opportunity, Mapping):
            return opportunity.get(name, default)
        return getattr(opportunity, name, default)

    strategy = str(_field("strategy") or "").strip().lower()
    title = str(_field("title") or "").strip().lower()
    positions_raw = _field("positions_to_take", [])
    positions = [p for p in positions_raw if isinstance(p, Mapping)] if isinstance(positions_raw, list) else []

    if strategy == "stat_arb":
        if title.startswith("certainty shock:"):
            return "certainty_shock"
        if title.startswith("temporal decay:"):
            return "decay_curve"

    if strategy in {"btc_eth_maker_quote", "btc_eth_directional_edge", "btc_eth_convergence"}:
        for pos in positions:
            if pos.get("_substrategy_metadata"):
                sub = _normalize_sub_strategy(str(pos.get("sub_strategy") or ""))
                if sub:
                    return sub
        return None

    if strategy == "news_edge":
        for pos in positions:
            payload = pos.get("_news_edge")
            if isinstance(payload, Mapping):
                direction = _normalize_sub_strategy(str(payload.get("direction") or ""))
                if direction:
                    return direction
        return None

    if strategy == "cross_platform":
        pm = next(
            (pos for pos in positions if str(pos.get("platform") or "").strip().lower() == "polymarket"),
            None,
        )
        kalshi = next(
            (pos for pos in positions if str(pos.get("platform") or "").strip().lower() == "kalshi"),
            None,
        )
        if pm and kalshi:
            pm_outcome = _normalize_sub_strategy(str(pm.get("outcome") or ""))
            kalshi_outcome = _normalize_sub_strategy(str(kalshi.get("outcome") or ""))
            if pm_outcome and kalshi_outcome:
                return f"poly_{pm_outcome}_kalshi_{kalshi_outcome}"
        return None

    if strategy == "settlement_lag":
        if title.startswith("settlement lag (negrisk):"):
            return "negrisk_bundle"
        if title.startswith("settlement lag:"):
            return "binary_market"
        return None

    if strategy == "negrisk":
        if title.startswith("negrisk short:"):
            return "binary_short"
        if title.startswith("negrisk:"):
            return "binary_long"
        if title.startswith("multi-outcome short:"):
            return "multi_outcome_short"
        if title.startswith("multi-outcome:"):
            return "multi_outcome_long"
        return None

    if strategy == "miracle":
        if title.startswith("stale market:"):
            return "stale_market"
        if title.startswith("miracle:"):
            return "impossibility_scan"
        return None

    return None


def _serialize_with_sub_strategy(opportunity: object) -> dict:
    payload = serialize_opportunity_with_links(opportunity)
    payload["strategy_subtype"] = _derive_opportunity_sub_strategy(opportunity)
    return payload


def _resolve_fastapi_param(value: Any) -> Any:
    if isinstance(value, Param):
        return value.default
    return value


def _coerce_float_param(value: Any, default: float) -> float:
    resolved = _resolve_fastapi_param(value)
    if resolved is None:
        return default
    try:
        return float(resolved)
    except Exception:
        return default


def _coerce_int_param(value: Any, default: int) -> int:
    resolved = _resolve_fastapi_param(value)
    if resolved is None:
        return default
    try:
        return int(resolved)
    except Exception:
        return default


async def _resolve_strategy_to_filter(strategy_param: Optional[str]) -> list[str]:
    """Resolve strategy param to list of strategy type strings.

    Accepts:
        - Built-in strategy type: "basic", "negrisk", etc.
        - Execution strategy key: "slug"
    """
    strategy_param = _resolve_fastapi_param(strategy_param)
    if not strategy_param:
        return []
    return [
        part
        for part in {piece.strip().lower() for piece in str(strategy_param).split(",")}
        if part
    ]


def _normalize_weather_edge_title(title: str) -> str:
    prefix = "weather edge:"
    return title[len(prefix) :].lstrip() if title.lower().startswith(prefix) else title


def _coerce_payload_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _payload_title(payload: Mapping[str, Any]) -> str:
    return _normalize_weather_edge_title(str(payload.get("title") or ""))


def _payload_search_matches(payload: Mapping[str, Any], search_lower: str) -> bool:
    if search_lower in _payload_title(payload).lower():
        return True
    event_title = str(payload.get("event_title") or "").lower()
    if event_title and search_lower in event_title:
        return True
    for market in payload.get("markets") or []:
        if not isinstance(market, Mapping):
            continue
        if search_lower in str(market.get("question") or "").lower():
            return True
    return False


def _sort_opportunity_payloads(
    payloads: list[dict[str, Any]],
    *,
    sort_by: Optional[str],
    sort_dir: Optional[str],
) -> None:
    reverse = (sort_dir or "desc") != "asc"
    effective_sort = sort_by or "ai_score"

    def _ai_analysis(payload: Mapping[str, Any]) -> Mapping[str, Any]:
        value = payload.get("ai_analysis")
        return value if isinstance(value, Mapping) else {}

    if effective_sort == "ai_score":
        payloads.sort(
            key=lambda payload: (
                bool(_ai_analysis(payload))
                and str(_ai_analysis(payload).get("recommendation") or "").strip().lower() != "pending",
                _coerce_payload_float(_ai_analysis(payload).get("overall_score"), 0.0),
                _coerce_payload_float(payload.get("roi_percent"), 0.0),
            ),
            reverse=reverse,
        )
        return

    if effective_sort == "profit":
        payloads.sort(key=lambda payload: _coerce_payload_float(payload.get("net_profit"), 0.0), reverse=reverse)
        return

    if effective_sort == "liquidity":
        payloads.sort(key=lambda payload: _coerce_payload_float(payload.get("min_liquidity"), 0.0), reverse=reverse)
        return

    if effective_sort == "risk":
        payloads.sort(key=lambda payload: _coerce_payload_float(payload.get("risk_score"), 0.0), reverse=reverse)
        return

    if effective_sort == "roi":
        payloads.sort(
            key=lambda payload: (
                str(_ai_analysis(payload).get("recommendation") or "").strip().lower() in {"skip", "strong_skip"},
                -_coerce_payload_float(payload.get("roi_percent"), 0.0)
                if reverse
                else _coerce_payload_float(payload.get("roi_percent"), 0.0),
            ),
        )


async def _list_market_opportunity_payload_page_fast(
    session: AsyncSession,
    *,
    min_profit: float,
    max_risk: float,
    strategy: Optional[str],
    min_liquidity: float,
    search: Optional[str],
    category: Optional[str],
    sort_by: Optional[str],
    sort_dir: Optional[str],
    exclude_strategy: Optional[str],
    sub_strategy: Optional[str],
    limit: int,
    offset: int,
    source: Literal["markets", "traders", "all"],
) -> Optional[tuple[int, list[dict[str, Any]]]]:
    payloads = await _list_filtered_opportunity_payloads(
        session,
        min_profit=min_profit,
        max_risk=max_risk,
        strategy=strategy,
        min_liquidity=min_liquidity,
        search=search,
        category=category,
        sort_by=sort_by,
        sort_dir=sort_dir,
        exclude_strategy=exclude_strategy,
        sub_strategy=sub_strategy,
        source=source,
    )
    total = len(payloads)
    page_window = max(limit, limit * OPPORTUNITY_PAGE_PREFETCH_MULTIPLIER)
    return total, payloads[offset : offset + page_window]


async def _get_market_opportunity_counts_fast(
    session: AsyncSession,
    *,
    min_profit: float,
    max_risk: float,
    min_liquidity: float,
    search: Optional[str],
    strategy: Optional[str],
    sub_strategy: Optional[str],
    category: Optional[str],
    source: Literal["markets", "traders", "all"],
) -> Optional[dict[str, Any]]:
    payloads = await _list_filtered_opportunity_payloads(
        session,
        min_profit=min_profit,
        max_risk=max_risk,
        strategy=strategy,
        min_liquidity=min_liquidity,
        search=search,
        category=category,
        sort_by=None,
        sort_dir=None,
        exclude_strategy=None,
        sub_strategy=sub_strategy,
        source=source,
        sort_results=False,
    )
    strategy_counts: dict[str, int] = {}
    category_counts: dict[str, int] = {}
    sub_strategy_counts: dict[str, int] = {}
    for payload in payloads:
        strategy_name = str(payload.get("strategy") or "").strip()
        if strategy_name:
            strategy_counts[strategy_name] = strategy_counts.get(strategy_name, 0) + 1
        category_name = str(payload.get("category") or "").strip().lower()
        if category_name:
            category_counts[category_name] = category_counts.get(category_name, 0) + 1
        subtype = _derive_opportunity_sub_strategy(payload)
        if subtype:
            sub_strategy_counts[subtype] = sub_strategy_counts.get(subtype, 0) + 1
    return {
        "strategies": strategy_counts,
        "categories": category_counts,
        "sub_strategies": sub_strategy_counts,
    }


async def _read_market_opportunity_payloads(session: AsyncSession) -> list[dict[str, Any]]:
    return await shared_state.read_active_opportunity_payloads(session)


async def _read_trader_opportunity_payloads(session: AsyncSession) -> list[dict[str, Any]]:
    result = await session.execute(
        select(ScannerSnapshot.opportunities_json).where(ScannerSnapshot.id == shared_state.TRADERS_SNAPSHOT_ID)
    )
    raw_payloads = result.scalar_one_or_none() or []
    payloads: list[dict[str, Any]] = []
    for item in raw_payloads:
        if not isinstance(item, Mapping):
            continue
        payload = dict(item)
        strategy_context = payload.get("strategy_context")
        context = dict(strategy_context) if isinstance(strategy_context, Mapping) else {}
        if str(context.get("source_key") or "").strip().lower() != "traders":
            context["source_key"] = "traders"
            payload["strategy_context"] = context
        payloads.append(payload)
    return payloads


async def _read_opportunity_payloads(
    session: AsyncSession,
    *,
    source: Literal["markets", "traders", "all"],
) -> list[dict[str, Any]]:
    source_key = _resolve_fastapi_param(source) or "markets"
    if source_key == "traders":
        return await _read_trader_opportunity_payloads(session)
    if source_key == "all":
        market_payloads = await _read_market_opportunity_payloads(session)
        trader_payloads = await _read_trader_opportunity_payloads(session)
        return market_payloads + trader_payloads
    return await _read_market_opportunity_payloads(session)


async def _list_filtered_opportunity_payloads(
    session: AsyncSession,
    *,
    min_profit: float,
    max_risk: float,
    strategy: Optional[str],
    min_liquidity: float,
    search: Optional[str],
    category: Optional[str],
    sort_by: Optional[str],
    sort_dir: Optional[str],
    exclude_strategy: Optional[str],
    sub_strategy: Optional[str],
    source: Literal["markets", "traders", "all"],
    sort_results: bool = True,
) -> list[dict[str, Any]]:
    min_profit = _coerce_float_param(min_profit, 0.0)
    max_risk = _coerce_float_param(max_risk, 1.0)
    min_liquidity = _coerce_float_param(min_liquidity, 0.0)
    strategy = _resolve_fastapi_param(strategy)
    search = _resolve_fastapi_param(search)
    category = _resolve_fastapi_param(category)
    sort_by = _resolve_fastapi_param(sort_by)
    sort_dir = _resolve_fastapi_param(sort_dir) or "desc"
    exclude_strategy = _resolve_fastapi_param(exclude_strategy)
    sub_strategy = _resolve_fastapi_param(sub_strategy)
    source = _resolve_fastapi_param(source) or "markets"

    strategies = set(await _resolve_strategy_to_filter(strategy))
    if not hasattr(session, "execute"):
        legacy_filter = OpportunityFilter(
            min_profit=min_profit,
            max_risk=max_risk,
            strategies=sorted(strategies),
            min_liquidity=min_liquidity,
            category=category,
        )
        legacy_opportunities = await shared_state.get_opportunities_from_db(
            session,
            legacy_filter,
            source=source,
        )
        payloads = [opportunity.model_dump() for opportunity in legacy_opportunities]
        if exclude_strategy:
            exclude_strategy_lower = str(exclude_strategy).strip().lower()
            payloads = [
                payload
                for payload in payloads
                if str(payload.get("strategy") or "").strip().lower() != exclude_strategy_lower
            ]
        if search:
            search_lower = str(search).strip().lower()
            payloads = [payload for payload in payloads if _payload_search_matches(payload, search_lower)]
        normalized_sub = _normalize_sub_strategy(sub_strategy)
        if normalized_sub:
            payloads = [
                payload
                for payload in payloads
                if _derive_opportunity_sub_strategy(payload) == normalized_sub
            ]
        if sort_results:
            _sort_opportunity_payloads(payloads, sort_by=sort_by, sort_dir=sort_dir)
        return payloads
    payloads = await _read_opportunity_payloads(session, source=source)

    if min_profit > 0:
        payloads = [
            payload
            for payload in payloads
            if _coerce_payload_float(payload.get("roi_percent"), 0.0) >= min_profit
        ]

    if max_risk < 1.0:
        payloads = [
            payload
            for payload in payloads
            if _coerce_payload_float(payload.get("risk_score"), 1.0) <= max_risk
        ]

    if strategies:
        payloads = [payload for payload in payloads if str(payload.get("strategy") or "").strip().lower() in strategies]

    if min_liquidity > 0:
        payloads = [
            payload
            for payload in payloads
            if _coerce_payload_float(payload.get("min_liquidity"), 0.0) >= min_liquidity
        ]

    if category:
        category_lower = str(category).strip().lower()
        payloads = [
            payload
            for payload in payloads
            if str(payload.get("category") or "").strip().lower() == category_lower
        ]

    if exclude_strategy:
        exclude_strategy_lower = str(exclude_strategy).strip().lower()
        payloads = [
            payload
            for payload in payloads
            if str(payload.get("strategy") or "").strip().lower() != exclude_strategy_lower
        ]

    if search:
        search_lower = str(search).strip().lower()
        payloads = [payload for payload in payloads if _payload_search_matches(payload, search_lower)]

    normalized_sub = _normalize_sub_strategy(sub_strategy)
    if normalized_sub:
        payloads = [
            payload
            for payload in payloads
            if _derive_opportunity_sub_strategy(payload) == normalized_sub
        ]

    if sort_results:
        _sort_opportunity_payloads(payloads, sort_by=sort_by, sort_dir=sort_dir)

    return payloads


async def _rollback_session_if_needed(session: AsyncSession) -> None:
    try:
        if session.in_transaction():
            await session.rollback()
    except Exception:
        pass


async def _hydrate_opportunity_payloads(
    session: AsyncSession,
    payloads: list[dict[str, Any]],
    *,
    source: Literal["markets", "traders", "all"],
    include_live_prices: bool = False,
    include_price_history: bool = False,
    block_for_history_backfill: bool = False,
) -> list[Opportunity]:
    opportunities: list[Opportunity] = []
    for payload in payloads:
        try:
            item = dict(payload)
            item["title"] = _payload_title(item)
            opportunities.append(Opportunity.model_validate(item))
        except Exception:
            continue

    await _rollback_session_if_needed(session)

    if not include_price_history:
        for opportunity in opportunities:
            for market in opportunity.markets:
                if isinstance(market, dict):
                    market.pop("price_history", None)

    source_key = _resolve_fastapi_param(source) or "markets"
    if opportunities and include_live_prices and source_key in {"markets", "all"}:
        try:
            from services.scanner import scanner as market_scanner

            opportunities = await market_scanner.refresh_opportunity_prices(
                opportunities,
                drop_stale=False,
            )
        except Exception:
            pass

    if not opportunities:
        return []

    if not include_price_history:
        return opportunities

    missing_history = []
    for opportunity in opportunities:
        if any(
            not isinstance(market.get("price_history"), list) or len(market.get("price_history") or []) < 2
            for market in opportunity.markets
            if isinstance(market, dict)
        ):
            missing_history.append(opportunity)

    if missing_history:
        try:
            from services.scanner import scanner as market_scanner

            await market_scanner.attach_price_history_to_opportunities(
                missing_history,
                timeout_seconds=2.0 if block_for_history_backfill else 0.0,
                block_for_backfill=block_for_history_backfill,
            )
        except Exception:
            pass

    return opportunities


# ==================== OPPORTUNITIES ====================


@router.get("/opportunities")
async def get_opportunities(
    response: Response,
    session: AsyncSession = Depends(get_db_session),
    min_profit: float = Query(0.0, description="Minimum profit percentage"),
    max_risk: float = Query(1.0, description="Maximum risk score (0-1)"),
    strategy: Optional[str] = Query(
        None,
        description="Filter by strategy type/slug (e.g. basic, negrisk, my_custom_strategy)",
    ),
    min_liquidity: float = Query(0.0, description="Minimum liquidity in USD"),
    search: Optional[str] = Query(None, description="Search query for market titles"),
    category: Optional[str] = Query(None, description="Filter by category (e.g., politics, sports, crypto)"),
    sort_by: Optional[str] = Query(
        None,
        description="Sort field: ai_score (default), roi, profit, liquidity, risk",
    ),
    sort_dir: Optional[str] = Query("desc", description="Sort direction: asc or desc"),
    exclude_strategy: Optional[str] = Query(
        None,
        description="Exclude a strategy type from results (e.g. btc_eth_maker_quote)",
    ),
    sub_strategy: Optional[str] = Query(
        None,
        description="Filter by strategy-specific subtype (e.g. certainty_shock, pure_arb)",
    ),
    limit: int = Query(50, description="Maximum results to return"),
    offset: int = Query(0, description="Number of results to skip"),
    source: Literal["markets", "traders", "all"] = Query(
        "markets",
        description="Opportunity source: markets, traders, all",
    ),
    include_price_history: bool = Query(
        False,
        description="Include per-market price history (for sparkline backfills).",
    ),
    block_for_history_backfill: bool = Query(
        False,
        description="Block briefly to backfill missing price history if needed.",
    ),
    validate_book: bool = Query(
        False,
        description=(
            "Annotate each returned opportunity with executable edge priced "
            "off the live order book (best ask per leg) instead of mid/last. "
            "Off by default (adds book fetches); use when you need the real, "
            "fillable edge rather than the mid-price ROI."
        ),
    ),
):
    """Get current arbitrage opportunities (from DB snapshot)."""
    total = 0
    page_payloads: list[dict[str, Any]] = []
    for attempt in range(DB_RETRY_ATTEMPTS):
        try:
            fast_path_result = await _list_market_opportunity_payload_page_fast(
                session,
                min_profit=min_profit,
                max_risk=max_risk,
                strategy=strategy,
                min_liquidity=min_liquidity,
                search=search,
                category=category,
                sort_by=sort_by,
                sort_dir=sort_dir,
                exclude_strategy=exclude_strategy,
                sub_strategy=sub_strategy,
                limit=limit,
                offset=offset,
                source=source,
            )
            if fast_path_result is not None:
                total, page_payloads = fast_path_result
            else:
                opportunity_payloads = await _list_filtered_opportunity_payloads(
                    session,
                    min_profit=min_profit,
                    max_risk=max_risk,
                    strategy=strategy,
                    min_liquidity=min_liquidity,
                    search=search,
                    category=category,
                    sort_by=sort_by,
                    sort_dir=sort_dir,
                    exclude_strategy=exclude_strategy,
                    sub_strategy=sub_strategy,
                    source=source,
                )
                total = len(opportunity_payloads)
                page_window = max(limit, limit * OPPORTUNITY_PAGE_PREFETCH_MULTIPLIER)
                page_payloads = opportunity_payloads[offset : offset + page_window]
            break
        except Exception as exc:
            if isinstance(exc, HTTPException):
                raise
            if not _is_retryable_db_error(exc):
                raise
            if attempt >= DB_RETRY_ATTEMPTS - 1:
                raise HTTPException(status_code=503, detail="Database is busy; please retry.") from exc
            try:
                if session.in_transaction():
                    await session.rollback()
            except Exception:
                pass
            try:
                await recover_pool()
            except Exception:
                pass
            await asyncio.sleep(_db_retry_delay(attempt))

    paginated = (
        await _hydrate_opportunity_payloads(
            session,
            page_payloads,
            source=source,
            include_live_prices=False,
            include_price_history=include_price_history,
            block_for_history_backfill=block_for_history_backfill,
        )
    )[:limit]

    # Set total count header and return serialised list directly.
    # Using Response injection (not JSONResponse) lets FastAPI handle
    # content-negotiation and CORS headers correctly.
    serialized = [_serialize_with_sub_strategy(o) for o in paginated]
    if validate_book:
        await _annotate_executable_edge(serialized)
    response.headers["X-Total-Count"] = str(total)
    return serialized


async def _annotate_executable_edge(
    opportunities: list[dict[str, Any]],
    *,
    max_markets: int = 30,
) -> None:
    """Re-price opportunities against the live order book in place.

    The scanner computes ROI from mid/last prices, which overstates edge —
    most apparent arbs evaporate at the executable ask (the bid/ask spread on
    these markets routinely exceeds the mispricing). For each binary
    opportunity (two CLOB tokens) this adds an ``executable_edge`` block with
    the real best-ask per leg, the pair cost, the surviving edge %, and the
    thinner side's USD depth, so callers act on fillable edge, not illusions.
    """
    from services.ws_feeds import get_feed_manager

    feed = get_feed_manager()

    async def _ask(token_id: str):
        try:
            book = await feed.get_order_book(token_id)
        except Exception:
            return None, 0.0, False
        if not book or not getattr(book, "asks", None):
            return None, 0.0, False
        ask = book.asks[0].price
        depth = sum(lvl.price * lvl.size for lvl in book.asks[:5])
        return ask, depth, feed.is_fresh(token_id)

    count = 0
    for opp in opportunities:
        if count >= max_markets:
            break
        markets = opp.get("markets") or []
        if not markets:
            continue
        tokens = (markets[0] or {}).get("clob_token_ids") or []
        if len(tokens) != 2:
            continue
        count += 1
        (yes_ask, yes_depth, yes_fresh), (no_ask, no_depth, no_fresh) = await asyncio.gather(
            _ask(str(tokens[0])), _ask(str(tokens[1]))
        )
        if yes_ask is None or no_ask is None:
            opp["executable_edge"] = {"available": False}
            continue
        pair_cost = yes_ask + no_ask
        opp["executable_edge"] = {
            "available": True,
            "yes_ask": yes_ask,
            "no_ask": no_ask,
            "pair_cost": round(pair_cost, 4),
            "edge_pct": round((1.0 - pair_cost) * 100.0, 3),
            "min_depth_usd": round(min(yes_depth, no_depth), 2),
            "book_fresh": bool(yes_fresh and no_fresh),
        }


@router.get("/opportunities/ids")
async def get_opportunity_ids(
    session: AsyncSession = Depends(get_db_session),
    min_profit: float = Query(0.0, description="Minimum profit percentage"),
    max_risk: float = Query(1.0, description="Maximum risk score (0-1)"),
    strategy: Optional[str] = Query(
        None,
        description="Filter by strategy type/slug (e.g. basic, negrisk, my_custom_strategy)",
    ),
    min_liquidity: float = Query(0.0, description="Minimum liquidity in USD"),
    search: Optional[str] = Query(None, description="Search query for market titles"),
    category: Optional[str] = Query(None, description="Filter by category (e.g., politics, sports, crypto)"),
    sort_by: Optional[str] = Query(
        None,
        description="Sort field: ai_score (default), roi, profit, liquidity, risk",
    ),
    sort_dir: Optional[str] = Query("desc", description="Sort direction: asc or desc"),
    exclude_strategy: Optional[str] = Query(
        None,
        description="Exclude a strategy type from results (e.g. btc_eth_maker_quote)",
    ),
    sub_strategy: Optional[str] = Query(
        None,
        description="Filter by strategy-specific subtype (e.g. certainty_shock, pure_arb)",
    ),
    limit: int = Query(500, ge=1, le=2000, description="Maximum IDs to return"),
    offset: int = Query(0, ge=0, description="Number of IDs to skip"),
    source: Literal["markets", "traders", "all"] = Query(
        "markets",
        description="Opportunity source: markets, traders, all",
    ),
):
    limit = _coerce_int_param(limit, 500)
    offset = _coerce_int_param(offset, 0)
    opportunity_payloads = await _list_filtered_opportunity_payloads(
        session,
        min_profit=min_profit,
        max_risk=max_risk,
        strategy=strategy,
        min_liquidity=min_liquidity,
        search=search,
        category=category,
        sort_by=sort_by,
        sort_dir=sort_dir,
        exclude_strategy=exclude_strategy,
        sub_strategy=sub_strategy,
        source=source,
    )

    total = len(opportunity_payloads)
    paginated = opportunity_payloads[offset : offset + limit]

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "ids": [str(payload.get("id") or "") for payload in paginated if str(payload.get("id") or "").strip()],
    }


@router.get("/opportunities/search-polymarket")
async def search_polymarket_opportunities(
    q: str = Query(..., min_length=1, description="Search query for Polymarket and Kalshi markets"),
    limit: int = Query(50, ge=1, le=200, description="Maximum results to return"),
):
    """
    Fast keyword search across Polymarket and Kalshi.

    Returns matching markets as lightweight opportunity-shaped objects
    so the frontend can display them immediately.  Skips the expensive
    price-fetching and full arbitrage-detection pipeline to keep the
    response time under a few seconds.
    """

    try:
        # Read search settings to decide which platforms to query
        from api.routes_settings import get_or_create_settings

        settings = await get_or_create_settings()
        poly_enabled = getattr(settings, "search_polymarket_enabled", True)
        kalshi_enabled = getattr(settings, "search_kalshi_enabled", False)
        max_results = getattr(settings, "search_max_results", 50)
        limit = min(limit, max_results)

        # Build tasks for enabled platforms only
        tasks: dict[str, asyncio.Task] = {}
        if poly_enabled:
            tasks["poly"] = asyncio.ensure_future(polymarket_client.search_markets(q, limit=limit))
            tasks["poly_events"] = asyncio.ensure_future(polymarket_client.search_events(q, limit=limit))
        if kalshi_enabled:
            tasks["kalshi"] = asyncio.ensure_future(
                asyncio.wait_for(kalshi_client.search_events(q, limit=limit), timeout=5.0)
            )

        if not tasks:
            from fastapi.responses import JSONResponse
            response = JSONResponse(content=[])
            response.headers["X-Total-Count"] = "0"
            return response

        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        task_results = dict(zip(tasks.keys(), results))

        poly_markets = task_results.get("poly", [])
        poly_events = task_results.get("poly_events", [])
        kalshi_events = task_results.get("kalshi", [])

        # Handle errors gracefully (includes TimeoutError from Kalshi)
        if isinstance(poly_markets, BaseException):
            poly_markets = []
        if isinstance(poly_events, BaseException):
            poly_events = []
        if isinstance(kalshi_events, BaseException):
            kalshi_events = []

        # Collect markets from Polymarket events
        poly_event_markets = []
        for event in poly_events or []:
            poly_event_markets.extend(event.markets)

        # Collect Kalshi markets from events
        kalshi_markets = []
        for event in kalshi_events or []:
            kalshi_markets.extend(event.markets)

        # Merge all markets, dedup by condition_id
        seen_cids: set[str] = set()
        all_markets: list = []
        for m in list(poly_markets) + poly_event_markets + kalshi_markets:
            cid = m.condition_id or m.question[:80]
            if cid not in seen_cids:
                seen_cids.add(cid)
                all_markets.append(m)

        # Filter out expired markets
        now = datetime.now(timezone.utc)
        all_markets = [
            m
            for m in all_markets
            if bool(getattr(m, "active", True))
            and not bool(getattr(m, "closed", False))
            and (m.end_date is None or m.end_date > now)
        ]

        # Relevance filter: keep markets where the query appears in any
        # meaningful field (question, slug, event_slug, group_item_title,
        # or tags).  The Gamma API returns markets from matching *events*,
        # so we check broadly but still filter out completely unrelated
        # markets that happen to share an event.
        q_lower = q.lower()
        q_words = q_lower.split()

        def _market_relevant(m) -> bool:
            searchable = " ".join([
                m.question.lower(),
                (m.slug or "").lower().replace("-", " "),
                (m.event_slug or "").lower().replace("-", " "),
                (m.group_item_title or "").lower(),
                " ".join(t.lower() for t in (m.tags or [])),
            ])
            return any(w in searchable for w in q_words)

        all_markets = [m for m in all_markets if _market_relevant(m)]

        if not all_markets:
            from fastapi.responses import JSONResponse

            response = JSONResponse(content=[])
            response.headers["X-Total-Count"] = "0"
            return response

        # Build lightweight opportunity objects from matched markets
        # so the frontend can render them with the existing UI.
        results: list[dict] = []
        seen: set[str] = set()
        for market in all_markets:
            mid = market.condition_id or market.question[:80]
            if mid in seen:
                continue
            seen.add(mid)

            platform = getattr(market, "platform", "polymarket") or "polymarket"

            # Best-effort price from the market's own outcome data
            yes_price = market.yes_price
            no_price = market.no_price

            slug = market.slug or ""
            event_slug = getattr(market, "event_slug", "") or ""
            category = ""
            volume = float(market.volume or 0)
            liquidity = float(getattr(market, "liquidity", 0) or market.volume or 0)

            results.append(
                attach_market_links_to_opportunity_dict(
                    {
                        "id": f"search-{mid}",
                        "stable_id": f"search-{mid}",
                        "title": market.question,
                        "description": (f"{platform.title()} market — Yes {yes_price:.0%} / No {no_price:.0%}"),
                        "event_title": market.question,
                        "event_slug": event_slug,
                        "strategy": "search",
                        "total_cost": 0.0,
                        "expected_payout": 0.0,
                        "gross_profit": 0.0,
                        "fee": 0.0,
                        "net_profit": 0.0,
                        "roi_percent": 0.0,
                        "risk_score": 0.0,
                        "risk_factors": [],
                        "min_liquidity": liquidity,
                        "volume": volume,
                        "max_position_size": 0.0,
                        "category": category,
                        "detected_at": datetime.now(timezone.utc).isoformat(),
                        "expires_at": (market.end_date.isoformat() if market.end_date else None),
                        "resolution_date": (market.end_date.isoformat() if market.end_date else None),
                        "platform": platform,
                        "positions_to_take": [],
                        "markets": [
                            {
                                "id": market.condition_id or "",
                                "condition_id": market.condition_id or "",
                                "question": market.question,
                                "slug": slug,
                                "event_slug": event_slug,
                                "platform": platform,
                                "yes_price": yes_price,
                                "no_price": no_price,
                                "volume": volume,
                                "liquidity": liquidity,
                            }
                        ],
                        "ai_analysis": None,
                    }
                )
            )

        results = results[:limit]

        from fastapi.responses import JSONResponse

        response = JSONResponse(content=results)
        response.headers["X-Total-Count"] = str(len(results))
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/opportunities/search-polymarket/evaluate")
async def evaluate_search_markets(
    body: dict = {},
    session: AsyncSession = Depends(get_db_session),
):
    """
    Trigger strategy evaluation on search result markets.

    Requests a scan from the scanner worker (writes to DB); results
    will appear once the worker runs and updates the snapshot.
    """
    condition_ids = body.get("condition_ids", [])

    await shared_state.request_one_scan(session, condition_ids=condition_ids or None)

    return {
        "status": "evaluating",
        "count": len(condition_ids),
        "message": (
            "Strategy scan requested"
            + (f" for {len(condition_ids)} market(s)" if condition_ids else "")
            + ". Detected opportunities will appear in the Markets tab."
        ),
    }


@router.get("/opportunities/counts")
async def get_opportunity_counts(
    session: AsyncSession = Depends(get_db_session),
    min_profit: float = Query(0.0, description="Minimum profit percentage"),
    max_risk: float = Query(1.0, description="Maximum risk score (0-1)"),
    min_liquidity: float = Query(0.0, description="Minimum liquidity in USD"),
    search: Optional[str] = Query(None, description="Search query for market titles"),
    strategy: Optional[str] = Query(
        None,
        description="Optional strategy type/slug filter for subfilter lookups",
    ),
    sub_strategy: Optional[str] = Query(
        None,
        description="Optional strategy subtype filter (e.g. certainty_shock, pure_arb)",
    ),
    category: Optional[str] = Query(None, description="Optional category filter for subfilter lookups"),
    source: Literal["markets", "traders", "all"] = Query(
        "markets",
        description="Opportunity source: markets, traders, all",
    ),
):
    """Get counts of opportunities grouped by strategy and category.

    Applies base filters (profit, risk, liquidity, search). Optional strategy/category
    filters can be supplied to narrow subtype counts for strategy-specific subfilters.
    """
    opportunity_payloads: list[dict[str, Any]] = []
    for attempt in range(DB_RETRY_ATTEMPTS):
        try:
            fast_counts = await _get_market_opportunity_counts_fast(
                session,
                min_profit=min_profit,
                max_risk=max_risk,
                min_liquidity=min_liquidity,
                search=search,
                strategy=strategy,
                sub_strategy=sub_strategy,
                category=category,
                source=source,
            )
            if fast_counts is not None:
                return fast_counts
            opportunity_payloads = await _list_filtered_opportunity_payloads(
                session,
                min_profit=min_profit,
                max_risk=max_risk,
                strategy=strategy,
                min_liquidity=min_liquidity,
                search=search,
                category=category,
                sort_by=None,
                sort_dir=None,
                exclude_strategy=None,
                sub_strategy=sub_strategy,
                source=source,
                sort_results=False,
            )
            break
        except Exception as exc:
            if isinstance(exc, HTTPException):
                raise
            if not _is_retryable_db_error(exc):
                raise
            if attempt >= DB_RETRY_ATTEMPTS - 1:
                raise HTTPException(status_code=503, detail="Database is busy; please retry.") from exc
            try:
                if session.in_transaction():
                    await session.rollback()
            except Exception:
                pass
            try:
                await recover_pool()
            except Exception:
                pass
            await asyncio.sleep(_db_retry_delay(attempt))

    # Count by strategy
    strategy_counts: dict[str, int] = {}
    category_counts: dict[str, int] = {}
    sub_strategy_counts: dict[str, int] = {}
    for payload in opportunity_payloads:
        strategy_name = str(payload.get("strategy") or "").strip()
        if strategy_name:
            strategy_counts[strategy_name] = strategy_counts.get(strategy_name, 0) + 1
        category_name = str(payload.get("category") or "").strip().lower()
        if category_name:
            cat = category_name
            category_counts[cat] = category_counts.get(cat, 0) + 1
        sub = _derive_opportunity_sub_strategy(payload)
        if sub:
            sub_strategy_counts[sub] = sub_strategy_counts.get(sub, 0) + 1

    return {
        "strategies": strategy_counts,
        "categories": category_counts,
        "sub_strategies": sub_strategy_counts,
    }


@router.get("/opportunities/{opportunity_id}", response_model=Opportunity)
async def get_opportunity(
    opportunity_id: str,
    source: Literal["markets", "traders", "all"] = Query(
        "markets",
        description="Opportunity source: markets, traders, all",
    ),
    session: AsyncSession = Depends(get_db_session),
):
    """Get a specific opportunity by ID"""
    payloads = await _read_opportunity_payloads(session, source=source)
    matching_payload = next((payload for payload in payloads if str(payload.get("id") or "") == opportunity_id), None)
    if matching_payload is not None:
        opportunities = await _hydrate_opportunity_payloads(
            session,
            [matching_payload],
            source=source,
            include_live_prices=True,
            include_price_history=True,
            block_for_history_backfill=True,
        )
        if opportunities:
            return _serialize_with_sub_strategy(opportunities[0])
    raise HTTPException(status_code=404, detail="Opportunity not found")


@router.post("/scan")
async def trigger_scan(session: AsyncSession = Depends(get_db_session)):
    """Manually request one scan. Scanner worker will run a scan on its next loop."""
    if global_pause_state.is_paused:
        raise HTTPException(
            status_code=409,
            detail="Global pause is active. Resume all workers before requesting a scan.",
        )
    await shared_state.request_one_scan(session)
    return {
        "status": "started",
        "timestamp": utcnow().isoformat(),
        "message": "Scan requested; scanner worker will run on next cycle.",
    }


# ==================== SCANNER STATUS ====================


@router.get("/scanner/status")
async def get_scanner_status(session: AsyncSession = Depends(get_db_session)):
    """Get scanner status (from DB snapshot)."""
    return await shared_state.get_scanner_status_from_db(session)


@router.post("/scanner/start")
async def start_scanner(session: AsyncSession = Depends(get_db_session)):
    """Start/resume the scanner (worker reads control from DB)."""
    if global_pause_state.is_paused:
        raise HTTPException(
            status_code=409,
            detail="Global pause is active. Use /workers/resume-all first.",
        )
    await shared_state.set_scanner_paused(session, False)
    return {"status": "started", **await shared_state.get_scanner_status_from_db(session)}


@router.post("/scanner/pause")
async def pause_scanner(session: AsyncSession = Depends(get_db_session)):
    """Pause the scanner (worker skips scans when paused)."""
    await shared_state.set_scanner_paused(session, True)
    return {"status": "paused", **await shared_state.get_scanner_status_from_db(session)}


@router.post("/scanner/interval")
async def set_scanner_interval(
    session: AsyncSession = Depends(get_db_session),
    interval_seconds: int = Query(..., ge=10, le=3600),
):
    """Set the scan interval (10-3600 seconds)."""
    await shared_state.set_scanner_interval(session, interval_seconds)
    return {"status": "updated", **await shared_state.get_scanner_status_from_db(session)}


# ==================== OPPORTUNITIES CLEANUP ====================


@router.delete("/opportunities")
async def clear_opportunities(session: AsyncSession = Depends(get_db_session)):
    """
    Clear all opportunities in the snapshot. Repopulated on next scanner run.
    """
    count = await shared_state.clear_opportunities_in_snapshot(session)
    return {
        "status": "success",
        "cleared_count": count,
        "message": f"Cleared {count} opportunities. Next scan will repopulate.",
    }


@router.post("/opportunities/cleanup")
async def cleanup_opportunities(
    session: AsyncSession = Depends(get_db_session),
    remove_expired: bool = Query(True, description="Remove opportunities past resolution date"),
    max_age_minutes: Optional[int] = Query(
        None, ge=1, le=1440, description="Remove opportunities older than X minutes"
    ),
):
    """
    Clean up stale opportunities in the snapshot.
    """
    results = await shared_state.cleanup_snapshot_opportunities(
        session, remove_expired=remove_expired, max_age_minutes=max_age_minutes
    )
    return {"status": "success", **results}


# ==================== WALLETS ====================


@router.get("/wallets")
async def get_tracked_wallets():
    """Get all tracked wallets"""
    return await wallet_tracker.get_all_wallets()


@router.post("/wallets")
async def add_wallet(address: str, label: Optional[str] = None):
    """Add a wallet to track"""
    try:
        resolved_address = await _resolve_wallet_address(address)
        await wallet_tracker.add_wallet(resolved_address, label, fetch_initial=False)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "success", "address": resolved_address, "label": label}


@router.delete("/wallets/{address}")
async def remove_wallet(address: str):
    """Remove a tracked wallet"""
    await wallet_tracker.remove_wallet(address)
    return {"status": "success", "address": address}


@router.get("/wallets/{address}")
async def get_wallet_info(address: str):
    """Get info for a specific wallet"""
    resolved_address = await _resolve_wallet_address(address)
    info = await wallet_tracker.get_wallet_info(resolved_address)
    if not info:
        raise HTTPException(status_code=404, detail="Wallet not found")
    return info


@router.get("/wallets/{address}/positions")
async def get_wallet_positions(address: str, include_prices: bool = True):
    """Get current positions for a wallet with optional current market prices"""
    try:
        resolved_address = await _resolve_wallet_address(address)
        if include_prices:
            positions = await polymarket_client.get_wallet_positions_with_prices(resolved_address)
        else:
            positions = await polymarket_client.get_wallet_positions(resolved_address)
        return {"address": resolved_address, "positions": positions}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/wallets/{address}/profile")
async def get_wallet_profile(address: str):
    """Get profile information for a wallet (username, etc.)"""
    try:
        resolved_address = await _resolve_wallet_address(address)
        profile = await polymarket_client.get_user_profile(resolved_address)
        profile["address"] = resolved_address
        return profile
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/wallets/{address}/trades")
async def get_wallet_trades(address: str, limit: int = 100):
    """Get recent trades for a wallet"""
    try:
        resolved_address = await _resolve_wallet_address(address)
        trades = await polymarket_client.get_wallet_trades(resolved_address, limit)
        return {"address": resolved_address, "trades": trades}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/wallets/recent-trades/all")
async def get_all_recent_trades(
    limit: int = Query(100, ge=1, le=500, description="Maximum trades to return"),
    hours: int = Query(24, ge=1, le=168, description="Only show trades from last N hours"),
):
    """
    Get recent trades from all tracked wallets, aggregated and sorted by timestamp.

    This provides a feed of the most recent/timely trading opportunities based on
    what wallets you're tracking are doing.
    """
    try:
        from datetime import timedelta

        wallets = await wallet_tracker.get_all_wallets()
        all_trades = []

        cutoff_time = utcnow() - timedelta(hours=hours)

        for wallet in wallets:
            wallet_address = wallet.get("address", "")
            wallet_label = wallet.get("label", wallet_address[:10] + "...")
            wallet_username = wallet.get("username") or ""
            recent_trades = wallet.get("recent_trades", [])

            for trade in recent_trades:
                # Parse timestamp - check multiple field names
                trade_time_str = (
                    trade.get("match_time")
                    or trade.get("timestamp")
                    or trade.get("time")
                    or trade.get("created_at")
                    or trade.get("createdAt")
                )
                if trade_time_str:
                    try:
                        if isinstance(trade_time_str, (int, float)):
                            trade_time = utcfromtimestamp(trade_time_str)
                        elif "T" in str(trade_time_str) or "-" in str(trade_time_str):
                            # Parse ISO format, normalizing to naive UTC
                            parsed = datetime.fromisoformat(str(trade_time_str).replace("Z", "+00:00"))
                            # Convert aware datetimes to UTC then strip tzinfo
                            if parsed.tzinfo is not None:
                                parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
                            trade_time = parsed
                        else:
                            trade_time = utcfromtimestamp(float(trade_time_str))

                        if trade_time < cutoff_time:
                            continue
                    except (ValueError, TypeError, OSError):
                        pass

                # Add wallet info to the trade
                enriched_trade = {
                    **trade,
                    "wallet_address": wallet_address,
                    "wallet_label": wallet_label,
                    "wallet_username": wallet_username,
                }
                all_trades.append(enriched_trade)

        # Enrich trades with market names and normalized timestamps
        from services.polymarket import polymarket_client

        all_trades = await polymarket_client.enrich_trades_with_market_info(all_trades)

        # Sort by normalized timestamp (most recent first)
        def _parse_to_naive_utc(raw_ts):
            """Parse a timestamp value to a naive UTC datetime."""
            if isinstance(raw_ts, (int, float)):
                try:
                    return utcfromtimestamp(raw_ts)
                except (ValueError, OSError):
                    return None
            if isinstance(raw_ts, str) and raw_ts.strip():
                text = raw_ts.strip()
                try:
                    if "T" in text or "-" in text:
                        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
                        if parsed.tzinfo is not None:
                            parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
                        return parsed
                    return utcfromtimestamp(float(text))
                except (ValueError, TypeError, OSError):
                    return None
            return None

        def get_sort_key(t):
            ts = t.get("timestamp_iso", "")
            if ts:
                parsed = _parse_to_naive_utc(ts)
                if parsed is not None:
                    return parsed
            # Fallback to raw timestamp fields
            raw = t.get("match_time") or t.get("timestamp") or t.get("time") or t.get("created_at") or ""
            parsed = _parse_to_naive_utc(raw)
            if parsed is not None:
                return parsed
            return datetime.min

        all_trades.sort(key=get_sort_key, reverse=True)

        # Limit results
        all_trades = all_trades[:limit]

        return {
            "trades": all_trades,
            "total": len(all_trades),
            "tracked_wallets": len(wallets),
            "hours_window": hours,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== MARKETS ====================


@router.get("/markets")
async def get_markets(active: bool = True, limit: int = 100, offset: int = 0):
    """Get markets from Polymarket"""
    try:
        markets = await polymarket_client.get_markets(active=active, limit=limit, offset=offset)
        return [m.model_dump() for m in markets]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/markets/{token_id}/orderbook")
async def get_market_orderbook(
    token_id: str,
    depth: int = Query(20, ge=1, le=200, description="Price levels per side to return"),
):
    """Live order book (bid/ask/depth) for a single CLOB token.

    Serves the same low-latency WebSocket book the execution engine sees,
    falling back to an HTTP fetch when the WS cache is stale. Returns best
    bid/ask, mid, spread (abs + bps) and cumulative depth in USD so callers
    can evaluate executable prices and size orders without flying blind.
    """
    from services.ws_feeds import get_feed_manager

    feed = get_feed_manager()
    try:
        book = await feed.get_order_book(token_id)
    except Exception as exc:  # pragma: no cover - network/runtime guard
        raise HTTPException(status_code=502, detail=f"order book fetch failed: {exc}")
    if book is None or (not book.bids and not book.asks):
        raise HTTPException(status_code=404, detail="No order book available for token")

    bids = [{"price": lvl.price, "size": lvl.size} for lvl in book.bids[:depth]]
    asks = [{"price": lvl.price, "size": lvl.size} for lvl in book.asks[:depth]]
    best_bid = book.bids[0].price if book.bids else None
    best_ask = book.asks[0].price if book.asks else None
    mid = spread = spread_bps = None
    if best_bid is not None and best_ask is not None:
        mid = (best_bid + best_ask) / 2.0
        spread = best_ask - best_bid
        if mid > 0:
            spread_bps = (spread / mid) * 10000.0

    return {
        "token_id": token_id,
        "is_fresh": feed.is_fresh(token_id),
        "best_bid": best_bid,
        "best_ask": best_ask,
        "mid": mid,
        "spread": spread,
        "spread_bps": spread_bps,
        "bid_depth_usd": sum(lvl.price * lvl.size for lvl in book.bids[:depth]),
        "ask_depth_usd": sum(lvl.price * lvl.size for lvl in book.asks[:depth]),
        "bids": bids,
        "asks": asks,
        "timestamp": utcnow().isoformat(),
    }


@router.get("/events")
async def get_events(closed: bool = False, limit: int = 100, offset: int = 0):
    """Get events from Polymarket"""
    try:
        events = await polymarket_client.get_events(closed=closed, limit=limit, offset=offset)
        return [e.model_dump() for e in events]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== STRATEGY INFO ====================


@router.get("/strategies")
async def get_strategies():
    """Get information about available strategies and plugins."""
    for attempt in range(DB_RETRY_ATTEMPTS):
        try:
            async with AsyncSessionLocal() as session:
                await ensure_system_opportunity_strategies_seeded(session)
                result = await session.execute(
                    select(Strategy)
                    .where(Strategy.enabled)
                    .order_by(
                        Strategy.is_system.desc(),
                        Strategy.sort_order.asc(),
                        Strategy.name.asc(),
                    )
                )
                plugins = result.scalars().all()

                db_entries: list[dict[str, object]] = [
                    {
                        "type": p.slug,
                        "name": p.name,
                        "description": p.description or f"Strategy: {p.slug}",
                        "is_plugin": not bool(p.is_system),
                        "is_system": bool(p.is_system),
                        "plugin_id": p.id,
                        "plugin_slug": p.slug,
                        "source_key": p.source_key or "scanner",
                        "status": p.status,
                        "enabled": bool(p.enabled),
                    }
                    for p in plugins
                ]
                combined: list[dict[str, object]] = db_entries

                # Deduplicate by type while preserving first occurrence ordering.
                seen_types: set[str] = set()
                deduped: list[dict[str, object]] = []
                for item in combined:
                    stype = str(item.get("type") or "").strip()
                    if not stype or stype in seen_types:
                        continue
                    seen_types.add(stype)
                    deduped.append(item)

                validation_profiles: dict[str, StrategyValidationProfile] = {}
                if seen_types:
                    rows = (
                        (
                            await session.execute(
                                select(StrategyValidationProfile).where(
                                    StrategyValidationProfile.strategy_type.in_(list(seen_types))
                                )
                            )
                        )
                        .scalars()
                        .all()
                    )
                    validation_profiles = {str(row.strategy_type): row for row in rows}

                enriched: list[dict[str, object]] = []
                for item in deduped:
                    strategy_type = str(item.get("type") or "")
                    is_plugin = bool(item.get("is_plugin"))
                    meta = _strategy_meta(strategy_type, is_plugin=is_plugin)
                    profile = validation_profiles.get(strategy_type)
                    enriched.append(
                        {
                            **item,
                            "domain": meta.get("domain"),
                            "timeframe": meta.get("timeframe"),
                            "sources": meta.get("sources"),
                            "validation_status": profile.status if profile is not None else "unknown",
                            "validation_sample_size": int(profile.sample_size or 0) if profile is not None else 0,
                        }
                    )
            return enriched
        except Exception as exc:
            if isinstance(exc, HTTPException):
                raise
            if not _is_retryable_db_error(exc):
                raise
            if attempt >= DB_RETRY_ATTEMPTS - 1:
                raise HTTPException(status_code=503, detail="Database is busy; please retry.") from exc
            try:
                await recover_pool()
            except Exception:
                pass
            await asyncio.sleep(_db_retry_delay(attempt))

    raise HTTPException(status_code=503, detail="Database is busy; please retry.")
