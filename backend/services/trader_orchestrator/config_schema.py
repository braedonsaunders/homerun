from __future__ import annotations

import ast
import logging
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import Strategy
from services.trader_orchestrator.sources.registry import (
    list_source_adapters,
    normalize_source_key,
)
from services.opportunity_strategy_catalog import (
    build_system_opportunity_strategy_rows,
)
from services.strategy_sdk import StrategySDK
from services.trader_orchestrator.templates import TRADER_TEMPLATES

logger = logging.getLogger(__name__)


def _normalize_strategy_source_key(value: Any) -> str:
    return normalize_source_key(value)


def _template_source_defaults(
    adapter_keys: set[str],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for template in TRADER_TEMPLATES:
        for source_config in list(template.get("source_configs") or []):
            source_key = _normalize_strategy_source_key(source_config.get("source_key"))
            if not source_key:
                continue
            if source_key not in adapter_keys:
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
    return build_system_opportunity_strategy_rows()


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
    adapter_keys: set[str],
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

        source_key = _normalize_strategy_source_key(row.source_key)
        if not source_key or source_key not in adapter_keys:
            continue
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
    adapter_keys = {adapter.key for adapter in adapters}

    source_defaults = _template_source_defaults(adapter_keys)
    strategy_rows = await _list_enabled_strategy_rows(session)
    strategies_by_source: dict[str, list[Any]] = {}
    for row in strategy_rows:
        source_key = _normalize_strategy_source_key(_row_value(row, "source_key"))
        if not source_key or source_key not in adapter_keys:
            continue
        strategies_by_source.setdefault(source_key, []).append(row)

    # Gather detection strategies (Strategy) that define evaluate().
    detection_strategies_by_source = await _list_detection_strategies_with_evaluate(session, adapter_keys)

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

        # 2. Detection strategies from Strategy table that have evaluate().
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
                or StrategySDK.trader_scope_defaults()
            )

        sources.append(
            {
                "key": adapter.key,
                "label": adapter.label,
                "description": adapter.description,
                "domains": adapter.domains,
                "signal_types": adapter.signal_types,
                "default_strategy_key": default_strategy_key,
                "strategy_options": strategy_options,
                "default_config": default_config,
                "scope_fields": StrategySDK.trader_scope_fields_schema() if adapter.key == "traders" else [],
            }
        )

    return {
        "version": "2026-02-21",
        "sources": sources,
        "shared_risk_fields": StrategySDK.trader_risk_fields_schema(),
        "shared_risk_defaults": StrategySDK.trader_risk_defaults(),
        "shared_exit_fields": [],
        "runtime_fields": StrategySDK.trader_runtime_fields_schema(),
        "default_runtime_metadata": StrategySDK.trader_runtime_defaults(),
        "trader_opportunity_filters_schema": StrategySDK.trader_opportunity_filter_config_schema(),
        "trader_opportunity_filters_defaults": StrategySDK.trader_opportunity_filter_defaults(),
        "copy_trading_schema": StrategySDK.copy_trading_config_schema(),
        "copy_trading_defaults": StrategySDK.copy_trading_defaults(),
    }
