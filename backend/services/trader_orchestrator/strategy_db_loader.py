from __future__ import annotations

import ast
import hashlib
import inspect
import sys
import traceback
import types
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import AsyncSessionLocal, Strategy
from services.trader_orchestrator.strategies.base import BaseTraderStrategy
from utils.utcnow import utcnow


# Modules DB strategy source is allowed to import.
ALLOWED_IMPORT_PREFIXES = {
    "__future__",
    # Strategy runtime APIs
    "services.trader_orchestrator.strategies.base",
    "services.trader_orchestrator.strategies",
    # Unified strategy base (BaseStrategy with evaluate/should_exit)
    "services.strategies.base",
    "services.strategies",
    # Platform services
    "services.ai",
    "services.ws_feeds",
    "services.chainlink_feed",
    "services.fee_model",
    "services.strategy_sdk",
    "config",
    "models",
    # Safe stdlib subset
    "math",
    "statistics",
    "collections",
    "dataclasses",
    "datetime",
    "enum",
    "functools",
    "hashlib",
    "itertools",
    "json",
    "logging",
    "operator",
    "re",
    "typing",
    "abc",
    "copy",
    "decimal",
    "fractions",
    "random",
    "bisect",
    "heapq",
    "textwrap",
    "uuid",
    "string",
}

BLOCKED_IMPORTS = {
    "os",
    "sys",
    "subprocess",
    "shutil",
    "pathlib",
    "importlib",
    "builtins",
    "ctypes",
    "socket",
    "http",
    "urllib",
    "requests",
    "httpx",
    "aiohttp",
    "pickle",
    "shelve",
    "marshal",
    "code",
    "codeop",
    "compile",
    "compileall",
    "exec",
    "eval",
    "__import__",
    "runpy",
    "dis",
    "signal",
    "threading",
    "multiprocessing",
    "concurrent",
    "asyncio",
    "io",
    "tempfile",
    "glob",
    "fnmatch",
    "webbrowser",
}

_BLOCKED_CALL_NAMES = {"exec", "eval", "compile", "__import__", "open", "input"}


class TraderStrategyValidationError(Exception):
    """Raised when DB strategy source code fails validation/load."""


@dataclass
class LoadedTraderStrategy:
    strategy_key: str
    source_key: str
    class_name: str
    instance: BaseTraderStrategy
    source_hash: str
    version: int
    module_name: str
    aliases: list[str]
    is_system: bool
    label: str
    description: str
    loaded_at: datetime


@dataclass
class StrategyAvailability:
    available: bool
    strategy_key: str
    resolved_key: str
    reason: Optional[str] = None


def _check_imports(tree: ast.AST) -> list[str]:
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in BLOCKED_IMPORTS:
                    violations.append(f"Blocked import: '{alias.name}' (line {node.lineno})")
                elif not any(alias.name.startswith(prefix) for prefix in ALLOWED_IMPORT_PREFIXES):
                    violations.append(
                        f"Disallowed import: '{alias.name}' (line {node.lineno}). "
                        "Only safe stdlib + trader strategy modules are allowed."
                    )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            root = module.split(".")[0]
            if root in BLOCKED_IMPORTS:
                violations.append(f"Blocked import: 'from {module}' (line {node.lineno})")
            elif module and not any(module.startswith(prefix) for prefix in ALLOWED_IMPORT_PREFIXES):
                violations.append(
                    f"Disallowed import: 'from {module}' (line {node.lineno}). "
                    "Only safe stdlib + trader strategy modules are allowed."
                )
    return violations


def _check_blocked_calls(tree: ast.AST) -> list[str]:
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            if node.func.id in _BLOCKED_CALL_NAMES:
                violations.append(f"Blocked call '{node.func.id}()' (line {node.lineno})")
            continue

        if isinstance(node.func, ast.Attribute):
            # Allow normal attribute calls like re.compile(). Only block
            # explicit builtins namespace usage (builtins.compile, etc.).
            if node.func.attr not in _BLOCKED_CALL_NAMES:
                continue
            target = node.func.value
            if isinstance(target, ast.Name) and target.id in {"builtins", "__builtins__"}:
                violations.append(f"Blocked call '{target.id}.{node.func.attr}()' (line {node.lineno})")
    return violations


def _find_class_def(tree: ast.AST, class_name: str | None) -> tuple[str | None, ast.ClassDef | None]:
    class_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    if class_name:
        for node in class_nodes:
            if node.name == class_name:
                return class_name, node
        return None, None

    # Auto-pick first class that might plausibly be a strategy candidate.
    for node in class_nodes:
        if node.name.lower().endswith("strategy"):
            return node.name, node
    if class_nodes:
        return class_nodes[0].name, class_nodes[0]
    return None, None


def validate_trader_strategy_source(source_code: str, class_name: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "valid": False,
        "class_name": None,
        "errors": [],
        "warnings": [],
    }

    try:
        tree = ast.parse(source_code)
    except SyntaxError as exc:
        result["errors"].append(f"Syntax error at line {exc.lineno}: {exc.msg}")
        return result

    violations = _check_imports(tree)
    violations.extend(_check_blocked_calls(tree))
    if violations:
        result["errors"].extend(violations)
        return result

    picked_name, class_node = _find_class_def(tree, class_name)
    if not picked_name or class_node is None:
        if class_name:
            result["errors"].append(f"Class '{class_name}' was not found in strategy source.")
        else:
            result["errors"].append("No class definition found in strategy source.")
        return result

    # Not requiring method body in AST because wrapper classes can inherit evaluate().
    # Unified strategies may have detect() instead of evaluate() — accept either.
    has_evaluate_method = any(
        isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == "evaluate"
        for item in class_node.body
    )
    has_detect_method = any(
        isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name in ("detect", "detect_async")
        for item in class_node.body
    )
    if not has_evaluate_method and not has_detect_method:
        result["warnings"].append(
            "Class does not define evaluate() or detect() directly; "
            "loader will validate inherited methods at runtime."
        )

    result["class_name"] = picked_name
    result["valid"] = True
    return result


class StrategyDBLoader:
    """Loads and caches DB-hosted trader strategy implementations."""

    def __init__(self) -> None:
        self._loaded: dict[str, LoadedTraderStrategy] = {}
        self._errors: dict[str, str] = {}
        self._aliases: dict[str, str] = {}
        self._module_counter = 0
        self._refresh_lock = None
        self._last_refresh_at: datetime | None = None

    @property
    def refresh_lock(self):
        if self._refresh_lock is None:
            import asyncio

            self._refresh_lock = asyncio.Lock()
        return self._refresh_lock

    def resolve_key(self, strategy_key: str) -> str:
        key = str(strategy_key or "").strip().lower()
        return self._aliases.get(key, key)

    def get_strategy(self, strategy_key: str) -> BaseTraderStrategy | None:
        resolved = self.resolve_key(strategy_key)
        loaded = self._loaded.get(resolved)
        return loaded.instance if loaded else None

    def get_availability(self, strategy_key: str) -> StrategyAvailability:
        resolved = self.resolve_key(strategy_key)
        if resolved in self._loaded:
            return StrategyAvailability(available=True, strategy_key=strategy_key, resolved_key=resolved)
        reason = self._errors.get(resolved)
        return StrategyAvailability(
            available=False,
            strategy_key=strategy_key,
            resolved_key=resolved,
            reason=reason or f"strategy_unavailable:{resolved}",
        )

    def list_strategy_keys(self) -> list[str]:
        return sorted(self._loaded.keys())

    def list_aliases(self) -> dict[str, str]:
        return dict(self._aliases)

    def get_runtime_status(self, strategy_key: str) -> dict[str, Any] | None:
        resolved = self.resolve_key(strategy_key)
        loaded = self._loaded.get(resolved)
        if loaded is not None:
            return {
                "strategy_key": loaded.strategy_key,
                "source_key": loaded.source_key,
                "class_name": loaded.class_name,
                "version": loaded.version,
                "source_hash": loaded.source_hash,
                "is_system": loaded.is_system,
                "label": loaded.label,
                "description": loaded.description,
                "aliases": list(loaded.aliases),
                "status": "loaded",
                "error_message": None,
                "loaded_at": loaded.loaded_at.isoformat(),
            }
        if resolved in self._errors:
            return {
                "strategy_key": resolved,
                "status": "error",
                "error_message": self._errors[resolved],
            }
        return None

    def _reset_modules_for_key(self, strategy_key: str) -> None:
        prefixes = (
            f"_trader_strategy_{strategy_key}_",
            f"_trader_strategy_alias_{strategy_key}_",
        )
        for module_name in list(sys.modules.keys()):
            if module_name.startswith(prefixes):
                sys.modules.pop(module_name, None)

    @staticmethod
    def _signature_valid(instance: BaseTraderStrategy) -> bool:
        evaluate = getattr(instance, "evaluate", None)
        if not callable(evaluate):
            return False
        try:
            sig = inspect.signature(evaluate)
        except (TypeError, ValueError):
            return False
        params = list(sig.parameters.values())
        if len(params) < 2:
            return False
        return True

    def _compile_and_load_row(self, row: Strategy) -> LoadedTraderStrategy:
        validation = validate_trader_strategy_source(row.source_code or "", row.class_name)
        if not validation.get("valid"):
            raise TraderStrategyValidationError("; ".join(validation.get("errors") or ["Validation failed"]))

        class_name = str(validation.get("class_name") or row.class_name or "").strip()
        if not class_name:
            raise TraderStrategyValidationError("Strategy class_name could not be resolved from source.")

        self._module_counter += 1
        module_name = f"_trader_strategy_{row.slug}_{self._module_counter}"

        try:
            code = compile(row.source_code, f"<trader-strategy:{row.slug}>", "exec")
            module = types.ModuleType(module_name)
            module.__file__ = f"<trader-strategy:{row.slug}>"
            module.__builtins__ = __builtins__
            sys.modules[module_name] = module
            exec(code, module.__dict__)  # noqa: S102

            strategy_class = getattr(module, class_name, None)
            if strategy_class is None:
                raise TraderStrategyValidationError(
                    f"Class '{class_name}' was not found after strategy source execution."
                )
            if not isinstance(strategy_class, type):
                raise TraderStrategyValidationError(f"'{class_name}' is not a class.")

            # Accept classes extending BaseTraderStrategy OR BaseStrategy (unified)
            from services.strategies.base import BaseStrategy

            if not issubclass(strategy_class, BaseTraderStrategy):
                if issubclass(strategy_class, BaseStrategy):
                    pass  # Accept unified strategies that extend BaseStrategy
                else:
                    raise TraderStrategyValidationError(
                        f"Class '{class_name}' must extend BaseTraderStrategy or BaseStrategy."
                    )

            instance = strategy_class()
            if not self._signature_valid(instance):
                raise TraderStrategyValidationError(
                    "Strategy evaluate() must support evaluate(signal, context)."
                )

            source_hash = hashlib.sha256((row.source_code or "").encode("utf-8")).hexdigest()[:16]
            aliases_raw = row.aliases if isinstance(row.aliases, list) else []
            aliases = [str(item).strip().lower() for item in aliases_raw if str(item).strip()]
            loaded = LoadedTraderStrategy(
                strategy_key=str(row.slug),
                source_key=str(row.source_key),
                class_name=class_name,
                instance=instance,
                source_hash=source_hash,
                version=int(row.version or 1),
                module_name=module_name,
                aliases=aliases,
                is_system=bool(row.is_system),
                label=str(row.name or row.slug),
                description=str(row.description or ""),
                loaded_at=datetime.now(timezone.utc),
            )
            return loaded
        except Exception:
            sys.modules.pop(module_name, None)
            raise

    async def refresh_from_db(
        self,
        *,
        session: AsyncSession | None = None,
        strategy_keys: list[str] | None = None,
    ) -> dict[str, Any]:
        owns_session = session is None
        if session is None:
            session = AsyncSessionLocal()

        keys_filter = {str(key or "").strip().lower() for key in (strategy_keys or []) if str(key or "").strip()}

        async with self.refresh_lock:
            try:
                query = select(Strategy).order_by(Strategy.slug.asc())
                if keys_filter:
                    query = query.where(Strategy.slug.in_(tuple(sorted(keys_filter))))
                rows = list((await session.execute(query)).scalars().all())

                next_loaded = dict(self._loaded)
                next_errors = dict(self._errors)
                next_aliases = dict(self._aliases)

                for row in rows:
                    strategy_key = str(row.slug or "").strip().lower()
                    self._reset_modules_for_key(strategy_key)
                    next_loaded.pop(strategy_key, None)
                    next_errors.pop(strategy_key, None)
                    next_aliases = {alias: key for alias, key in next_aliases.items() if key != strategy_key}

                    if not bool(row.enabled):
                        row.status = "disabled"
                        row.error_message = None
                        row.updated_at = utcnow()
                        continue

                    try:
                        loaded = self._compile_and_load_row(row)
                        next_loaded[strategy_key] = loaded
                        row.status = "loaded"
                        row.error_message = None
                        row.updated_at = utcnow()
                        for alias in loaded.aliases:
                            next_aliases[alias] = strategy_key
                    except Exception as exc:
                        error_message = str(exc)
                        tb = traceback.format_exc()
                        next_errors[strategy_key] = error_message
                        row.status = "error"
                        row.error_message = f"{error_message}\n{tb}"
                        row.updated_at = utcnow()

                self._loaded = next_loaded
                self._errors = next_errors
                self._aliases = next_aliases
                self._last_refresh_at = datetime.now(timezone.utc)

                await session.commit()

                return {
                    "loaded": sorted(self._loaded.keys()),
                    "errors": dict(self._errors),
                    "aliases": dict(self._aliases),
                    "refreshed_at": self._last_refresh_at.isoformat() if self._last_refresh_at else None,
                }
            finally:
                if owns_session:
                    await session.close()

    async def reload_strategy(self, strategy_key: str, *, session: AsyncSession | None = None) -> dict[str, Any]:
        key = str(strategy_key or "").strip().lower()
        if not key:
            raise ValueError("strategy_key is required")
        return await self.refresh_from_db(session=session, strategy_keys=[key])

    async def list_definitions(
        self,
        session: AsyncSession,
        *,
        source_key: str | None = None,
        enabled_only: bool = False,
    ) -> list[Strategy]:
        query = select(Strategy)
        if source_key:
            query = query.where(Strategy.source_key == str(source_key).strip().lower())
        if enabled_only:
            query = query.where(Strategy.enabled == True)  # noqa: E712
        query = query.order_by(Strategy.source_key.asc(), Strategy.slug.asc())
        return list((await session.execute(query)).scalars().all())


strategy_db_loader = StrategyDBLoader()


def serialize_trader_strategy_definition(row: Strategy) -> dict[str, Any]:
    runtime = strategy_db_loader.get_runtime_status(str(row.slug or ""))
    return {
        "id": row.id,
        "strategy_key": row.slug,
        "source_key": row.source_key,
        "label": row.name,
        "description": row.description,
        "class_name": row.class_name,
        "source_code": row.source_code,
        "default_params_json": row.config or {},
        "param_schema_json": row.config_schema or {},
        "aliases_json": row.aliases or [],
        "is_system": bool(row.is_system),
        "enabled": bool(row.enabled),
        "status": row.status,
        "error_message": row.error_message,
        "version": int(row.version or 1),
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        "runtime": runtime,
    }
