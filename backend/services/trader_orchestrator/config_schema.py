from __future__ import annotations

import ast
import logging
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import Strategy
from services.trader_orchestrator.sources.registry import (
    list_source_adapters,
    list_source_aliases,
)
from services.trader_orchestrator.strategy_catalog import (
    build_system_strategy_rows,
    ensure_system_trader_strategies_seeded,
)
from services.trader_orchestrator.templates import TRADER_TEMPLATES

logger = logging.getLogger(__name__)


_DEFAULT_METADATA = {
    "cadence_profile": "custom",
    "trading_window_utc": {"start": "00:00", "end": "23:59"},
    "tags": [],
    "notes": "",
    "resume_policy": "resume_full",
}

_SHARED_RISK_FIELDS: list[dict[str, Any]] = [
    {"key": "max_open_positions", "label": "Max Open Positions", "type": "integer", "min": 1, "max": 1000},
    {"key": "max_open_orders", "label": "Max Open Orders", "type": "integer", "min": 1, "max": 2000},
    {"key": "max_orders_per_cycle", "label": "Max Orders / Cycle", "type": "integer", "min": 1, "max": 1000},
    {"key": "max_trade_notional_usd", "label": "Max Trade Notional (USD)", "type": "number", "min": 1},
    {"key": "max_per_market_exposure_usd", "label": "Max Per-Market Exposure (USD)", "type": "number", "min": 1},
    {"key": "max_daily_loss_usd", "label": "Max Daily Loss (USD)", "type": "number", "min": 1},
    {"key": "cooldown_seconds", "label": "Cooldown (seconds)", "type": "integer", "min": 0},
]

_SHARED_EXIT_FIELDS: list[dict[str, Any]] = [
    {
        "key": "take_profit_pct",
        "label": "Take Profit (%)",
        "type": "number",
        "description": "Close position when profit exceeds this percentage",
        "required": False,
    },
    {
        "key": "stop_loss_pct",
        "label": "Stop Loss (%)",
        "type": "number",
        "description": "Close position when loss exceeds this percentage",
        "required": False,
    },
    {
        "key": "trailing_stop_pct",
        "label": "Trailing Stop (%)",
        "type": "number",
        "description": "Close if price falls this % below high water mark",
        "required": False,
    },
    {
        "key": "max_hold_minutes",
        "label": "Max Hold (minutes)",
        "type": "number",
        "description": "Force close after this many minutes",
        "required": False,
    },
    {
        "key": "min_hold_minutes",
        "label": "Min Hold (minutes)",
        "type": "number",
        "description": "Don't exit before this many minutes",
        "required": False,
    },
    {
        "key": "resolve_only",
        "label": "Resolve Only",
        "type": "boolean",
        "description": "Only close on market resolution (ignore TP/SL)",
        "required": False,
    },
]

_RUNTIME_FIELDS: list[dict[str, Any]] = [
    {
        "key": "resume_policy",
        "label": "Resume Policy",
        "type": "enum",
        "options": [
            {
                "value": "resume_full",
                "label": "Resume Full",
                "description": "Manage existing positions and continue processing new signals.",
            },
            {
                "value": "manage_only",
                "label": "Manage Existing Only",
                "description": "Manage/exit existing positions without opening new positions.",
            },
            {
                "value": "flatten_then_start",
                "label": "Flatten Then Start",
                "description": "Close existing positions first, then enable new entries.",
            },
        ],
    },
    {
        "key": "trading_window_utc",
        "label": "Trading Window (UTC)",
        "type": "object",
        "properties": [
            {"key": "start", "type": "time_hhmm"},
            {"key": "end", "type": "time_hhmm"},
        ],
    },
]

_TRADERS_SCOPE_FIELDS = [
    {
        "key": "traders_scope",
        "label": "Traders Scope",
        "type": "object",
        "properties": [
            {
                "key": "modes",
                "label": "Modes",
                "type": "array[string]",
                "options": ["tracked", "pool", "individual", "group"],
                "required": True,
            },
            {
                "key": "individual_wallets",
                "label": "Individual Wallets",
                "type": "array[string]",
                "required_if_mode": "individual",
            },
            {
                "key": "group_ids",
                "label": "Group IDs",
                "type": "array[string]",
                "required_if_mode": "group",
            },
        ],
    }
]


def _template_source_defaults() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for template in TRADER_TEMPLATES:
        for source_config in list(template.get("source_configs") or []):
            source_key = str(source_config.get("source_key") or "").strip().lower()
            if not source_key:
                continue
            if source_key not in out:
                out[source_key] = source_config
    return out


def _row_value(row: Any, key: str) -> Any:
    if isinstance(row, dict):
        return row.get(key)
    return getattr(row, key)


def _strategy_param_fields(row: Any) -> list[dict[str, Any]]:
    schema = _row_value(row, "config_schema")
    if not isinstance(schema, dict):
        return []
    fields = schema.get("param_fields")
    if isinstance(fields, list):
        return [field for field in fields if isinstance(field, dict)]
    return []


async def _list_enabled_strategy_rows(session: AsyncSession) -> list[Any]:
    await ensure_system_trader_strategies_seeded(session)
    rows = list(
        (
            await session.execute(
                select(Strategy)
                .where(Strategy.enabled == True)  # noqa: E712
                .order_by(
                    Strategy.source_key.asc(),
                    Strategy.slug.asc(),
                )
            )
        )
        .scalars()
        .all()
    )
    if rows:
        return rows
    # Cold-start fallback for environments that have not migrated/seeded yet.
    return build_system_strategy_rows()


def _detection_plugin_has_evaluate(source_code: str, class_name: str | None = None) -> bool:
    """Check if a detection strategy plugin defines a custom evaluate() method.

    Uses lightweight AST inspection rather than importing the module.
    """
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return False

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        # If class_name is specified, match it; otherwise check any BaseStrategy subclass.
        if class_name and node.name != class_name:
            continue
        is_base_strategy = any(
            (isinstance(b, ast.Name) and b.id == "BaseStrategy")
            or (isinstance(b, ast.Attribute) and b.attr == "BaseStrategy")
            for b in node.bases
        )
        if not is_base_strategy:
            continue
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == "evaluate":
                return True
    return False


async def _list_detection_strategies_with_evaluate(
    session: AsyncSession,
) -> dict[str, list[dict[str, Any]]]:
    """Query enabled Strategy rows and return those with evaluate() capability.

    Returns a dict keyed by source_key, each value a list of strategy option dicts.
    """
    try:
        rows = list(
            (
                await session.execute(
                    select(Strategy)
                    .where(Strategy.enabled == True)  # noqa: E712
                    .order_by(Strategy.source_key.asc(), Strategy.slug.asc())
                )
            )
            .scalars()
            .all()
        )
    except Exception as exc:
        logger.debug("Failed to query detection strategies for config schema: %s", exc)
        return {}

    by_source: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        source_code = row.source_code or ""
        class_name = row.class_name
        if not _detection_plugin_has_evaluate(source_code, class_name):
            continue

        source_key = str(row.source_key or "scanner").strip().lower()
        by_source.setdefault(source_key, []).append(
            {
                "key": row.slug,
                "label": f"{row.name} (detection)",
                "description": str(row.description or ""),
                "default_params": {},
                "param_fields": [],
                "status": str(row.status or "unknown"),
                "version": int(row.version or 1),
                "is_system": bool(row.is_system),
                "is_detection_strategy": True,
            }
        )
    return by_source


async def build_trader_config_schema(session: AsyncSession) -> dict[str, Any]:
    adapters = list_source_adapters()
    source_aliases = list_source_aliases()
    aliases_by_canonical: dict[str, list[str]] = {}
    for alias, canonical in source_aliases.items():
        aliases_by_canonical.setdefault(canonical, []).append(alias)

    source_defaults = _template_source_defaults()
    strategy_rows = await _list_enabled_strategy_rows(session)
    strategies_by_source: dict[str, list[Any]] = {}
    for row in strategy_rows:
        source_key = str(_row_value(row, "source_key") or "").strip().lower()
        if not source_key:
            continue
        strategies_by_source.setdefault(source_key, []).append(row)

    # Gather detection strategies (Strategy) that define evaluate().
    detection_strategies_by_source = await _list_detection_strategies_with_evaluate(session)

    sources: list[dict[str, Any]] = []
    for adapter in adapters:
        rows = strategies_by_source.get(adapter.key, [])
        strategy_options: list[dict[str, Any]] = []
        template_defaults = source_defaults.get(adapter.key, {})
        template_default_params = dict(template_defaults.get("strategy_params") or {})

        # 1. Dedicated trader strategies from Strategy table.
        for row in rows:
            key = str(_row_value(row, "slug") or "").strip().lower()
            if not key:
                continue
            row_defaults = _row_value(row, "config")
            default_params = dict(row_defaults or {}) if isinstance(row_defaults, dict) else {}
            if not default_params and template_default_params:
                default_params = dict(template_default_params)
            strategy_options.append(
                {
                    "key": key,
                    "label": str(_row_value(row, "name") or key),
                    "description": str(_row_value(row, "description") or ""),
                    "default_params": default_params,
                    "param_fields": _strategy_param_fields(row),
                    "status": str(_row_value(row, "status") or "unknown"),
                    "version": int(_row_value(row, "version") or 1),
                    "is_system": bool(_row_value(row, "is_system")),
                }
            )

        # 2. Passthrough option: use the detection strategy's evaluate() method.
        strategy_options.append(
            {
                "key": "__passthrough__",
                "label": "Passthrough (use detection strategy)",
                "description": (
                    "Delegates execution gating to the detection strategy's evaluate() method. "
                    "Uses the strategy that originally detected the opportunity."
                ),
                "default_params": {},
                "param_fields": [],
                "status": "active",
                "version": 1,
                "is_system": True,
                "is_passthrough": True,
            }
        )

        # 3. Detection strategies from Strategy table that have evaluate().
        existing_keys = {opt["key"] for opt in strategy_options}
        for det_opt in detection_strategies_by_source.get(adapter.key, []):
            if det_opt["key"] not in existing_keys:
                strategy_options.append(det_opt)

        default_strategy_key = strategy_options[0]["key"] if strategy_options else ""
        default_strategy_defaults = strategy_options[0]["default_params"] if strategy_options else {}
        default_config: dict[str, Any] = {
            "source_key": adapter.key,
            "strategy_key": default_strategy_key,
            "strategy_params": dict(template_default_params or default_strategy_defaults),
        }
        if adapter.key == "traders":
            default_config["traders_scope"] = dict(
                (source_defaults.get("traders") or {}).get("traders_scope")
                or {"modes": ["tracked", "pool"], "individual_wallets": [], "group_ids": []}
            )

        sources.append(
            {
                "key": adapter.key,
                "label": adapter.label,
                "description": adapter.description,
                "domains": adapter.domains,
                "signal_types": adapter.signal_types,
                "aliases": sorted(aliases_by_canonical.get(adapter.key, [])),
                "default_strategy_key": default_strategy_key,
                "strategy_options": strategy_options,
                "default_config": default_config,
                "scope_fields": list(_TRADERS_SCOPE_FIELDS if adapter.key == "traders" else []),
            }
        )

    return {
        "version": "2026-02-17",
        "source_aliases": source_aliases,
        "sources": sources,
        "shared_risk_fields": list(_SHARED_RISK_FIELDS),
        "shared_exit_fields": list(_SHARED_EXIT_FIELDS),
        "runtime_fields": list(_RUNTIME_FIELDS),
        "default_runtime_metadata": dict(_DEFAULT_METADATA),
    }
