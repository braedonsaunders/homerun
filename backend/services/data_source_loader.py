"""Unified loader/validator for DB-defined data sources."""

from __future__ import annotations

import ast
import hashlib
import re
import sys
import traceback
import types
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

from sqlalchemy import select

from utils.logger import get_logger

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = get_logger(__name__)


DATA_SOURCE_TEMPLATE = '''"""Data Source: My Custom Source

Implement a class extending BaseDataSource. Return normalized records from
fetch()/fetch_async() and optionally transform() each record.
"""

import httpx
from services.data_source_sdk import BaseDataSource


class MyCustomSource(BaseDataSource):
    name = "My Custom Source"
    description = "Fetches and normalizes external data"

    default_config = {
        "endpoint": "https://example.com/data",
        "limit": 100,
    }

    async def fetch_async(self):
        endpoint = str(self.config.get("endpoint") or "").strip()
        limit = int(self.config.get("limit") or 100)
        if not endpoint:
            return []

        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(endpoint)
            response.raise_for_status()
            payload = response.json()

        rows = payload if isinstance(payload, list) else payload.get("items", [])
        out = []
        for row in rows[:limit]:
            out.append({
                "external_id": str(row.get("id") or ""),
                "title": str(row.get("title") or "").strip(),
                "summary": str(row.get("summary") or "").strip(),
                "category": str(row.get("category") or "data").strip().lower(),
                "source": "custom_api",
                "url": row.get("url"),
                "observed_at": row.get("timestamp"),
                "payload": row,
                "geotagged": bool(row.get("lat") is not None and row.get("lon") is not None),
                "latitude": row.get("lat"),
                "longitude": row.get("lon"),
                "country_iso3": row.get("country_iso3"),
                "tags": ["custom", "api"],
            })
        return out

    def transform(self, item):
        # Optional second-stage transform hook per record.
        return item
'''


ALLOWED_IMPORT_PREFIXES = {
    "__future__",
    "services.data_source_sdk",
    "services.strategy_sdk",
    "services.news",
    "services.world_intelligence",
    "models",
    "utils",
    "config",
    "math",
    "statistics",
    "collections",
    "datetime",
    "re",
    "json",
    "random",
    "asyncio",
    "calendar",
    "pathlib",
    "decimal",
    "fractions",
    "itertools",
    "functools",
    "operator",
    "copy",
    "enum",
    "dataclasses",
    "typing",
    "abc",
    "time",
    "hashlib",
    "hmac",
    "base64",
    "uuid",
    "urllib.parse",
    "logging",
    "bisect",
    "heapq",
    "textwrap",
    "string",
    "concurrent",
    "httpx",
    "numpy",
    "scipy",
    "pandas",
    "sqlalchemy",
}


BLOCKED_IMPORTS = {
    "os",
    "sys",
    "subprocess",
    "shutil",
    "importlib",
    "builtins",
    "ctypes",
    "socket",
    "http",
    "urllib",
    "requests",
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
    "ast",
    "dis",
    "inspect",
    "signal",
    "multiprocessing",
    "io",
    "tempfile",
    "glob",
    "fnmatch",
    "webbrowser",
}


_BLOCKED_CALL_NAMES = {"exec", "eval", "compile", "__import__", "open", "input"}


class DataSourceValidationError(Exception):
    pass


def _check_imports(tree: ast.AST) -> list[str]:
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod_root = alias.name.split(".")[0]
                if alias.name.startswith("urllib.parse"):
                    continue
                if mod_root in BLOCKED_IMPORTS:
                    violations.append(f"Blocked import: '{alias.name}' (line {node.lineno})")
                elif not any(alias.name.startswith(p) for p in ALLOWED_IMPORT_PREFIXES):
                    violations.append(f"Disallowed import: '{alias.name}' (line {node.lineno})")
        elif isinstance(node, ast.ImportFrom):
            if not node.module:
                continue
            mod_root = node.module.split(".")[0]
            if node.module.startswith("urllib.parse"):
                continue
            if mod_root in BLOCKED_IMPORTS:
                violations.append(f"Blocked import: 'from {node.module}' (line {node.lineno})")
            elif not any(node.module.startswith(p) for p in ALLOWED_IMPORT_PREFIXES):
                violations.append(f"Disallowed import: 'from {node.module}' (line {node.lineno})")
    return violations


def _check_blocked_calls(tree: ast.AST) -> list[str]:
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            if node.func.id in _BLOCKED_CALL_NAMES:
                violations.append(f"Blocked call '{node.func.id}()' (line {node.lineno})")
        elif isinstance(node.func, ast.Attribute):
            if node.func.attr in _BLOCKED_CALL_NAMES:
                target = node.func.value
                if isinstance(target, ast.Name) and target.id in {"builtins", "__builtins__"}:
                    violations.append(f"Blocked call '{target.id}.{node.func.attr}()' (line {node.lineno})")
    return violations


def _find_source_class(tree: ast.AST, class_name: Optional[str] = None) -> Optional[str]:
    if class_name:
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                return class_name
        return None

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id == "BaseDataSource":
                return node.name
            if isinstance(base, ast.Attribute) and base.attr == "BaseDataSource":
                return node.name
    return None


def _extract_class_attribute(tree: ast.AST, class_name: str, attr_name: str) -> Optional[str]:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for stmt in node.body:
                if isinstance(stmt, ast.Assign):
                    for target in stmt.targets:
                        if isinstance(target, ast.Name) and target.id == attr_name:
                            if isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
                                return stmt.value.value
    return None


def _detect_capabilities(source_code: str) -> dict[str, bool]:
    has_fetch = bool(re.search(r"\bdef fetch\s*\(", source_code))
    has_fetch_async = bool(re.search(r"\basync\s+def fetch_async\s*\(", source_code))
    has_transform = bool(re.search(r"\bdef transform\s*\(", source_code))
    return {
        "has_fetch": has_fetch,
        "has_fetch_async": has_fetch_async,
        "has_transform": has_transform,
    }


def validate_data_source_source(source_code: str, class_name: Optional[str] = None) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    try:
        tree = ast.parse(source_code)
    except SyntaxError as exc:
        return {
            "valid": False,
            "errors": [f"Syntax error on line {exc.lineno}: {exc.msg}"],
            "warnings": [],
            "class_name": None,
            "source_name": None,
            "source_description": None,
            "capabilities": {"has_fetch": False, "has_fetch_async": False, "has_transform": False},
        }

    errors.extend(_check_imports(tree))
    errors.extend(_check_blocked_calls(tree))

    detected_class_name = _find_source_class(tree, class_name)
    if not detected_class_name:
        errors.append("No class extending BaseDataSource was found.")

    capabilities = _detect_capabilities(source_code)
    if not capabilities["has_fetch"] and not capabilities["has_fetch_async"]:
        errors.append("Data source must implement fetch() or fetch_async().")

    source_name = _extract_class_attribute(tree, detected_class_name, "name") if detected_class_name else None
    source_description = (
        _extract_class_attribute(tree, detected_class_name, "description")
        if detected_class_name
        else None
    )

    if not source_name:
        warnings.append("Class attribute 'name' is missing; UI will use slug fallback.")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "class_name": detected_class_name,
        "source_name": source_name,
        "source_description": source_description,
        "capabilities": capabilities,
    }


@dataclass
class DataSourceRuntime:
    slug: str
    class_name: str
    name: str
    description: str
    module: types.ModuleType
    instance: Any
    loaded_at: datetime
    source_hash: str
    run_count: int = 0
    error_count: int = 0
    last_run: datetime | None = None
    last_error: str | None = None


class DataSourceLoader:
    def __init__(self) -> None:
        self._runtimes: dict[str, DataSourceRuntime] = {}

    def load(
        self,
        slug: str,
        source_code: str,
        config: dict[str, Any] | None = None,
        class_name: str | None = None,
    ) -> DataSourceRuntime:
        normalized_slug = str(slug or "").strip().lower()
        if not normalized_slug:
            raise DataSourceValidationError("Source slug is required")

        validation = validate_data_source_source(source_code, class_name=class_name)
        if not validation["valid"]:
            raise DataSourceValidationError("; ".join(validation["errors"]))

        class_name_value = str(validation["class_name"])
        source_hash = hashlib.sha256(source_code.encode("utf-8")).hexdigest()
        module_name = f"_db_data_source_{normalized_slug}_{source_hash[:10]}"
        module = types.ModuleType(module_name)
        module.__file__ = f"<db-data-source:{normalized_slug}>"

        try:
            compiled = compile(source_code, module.__file__, "exec")
            exec(compiled, module.__dict__)
            source_cls = module.__dict__.get(class_name_value)
            if source_cls is None:
                raise DataSourceValidationError(f"Class '{class_name_value}' not found after compile")
            instance = source_cls()
            if not hasattr(instance, "configure"):
                raise DataSourceValidationError(f"Class '{class_name_value}' must implement configure(config)")
            instance.configure(config or {})
        except DataSourceValidationError:
            raise
        except Exception as exc:
            traceback_text = traceback.format_exc(limit=5)
            raise DataSourceValidationError(
                f"Failed to compile/load source '{normalized_slug}': {exc}\n{traceback_text}"
            ) from exc

        runtime = DataSourceRuntime(
            slug=normalized_slug,
            class_name=class_name_value,
            name=str(getattr(instance, "name", validation.get("source_name") or normalized_slug)),
            description=str(getattr(instance, "description", validation.get("source_description") or "")),
            module=module,
            instance=instance,
            loaded_at=datetime.now(timezone.utc),
            source_hash=source_hash,
        )

        self._runtimes[normalized_slug] = runtime
        return runtime

    def unload(self, slug: str) -> None:
        normalized_slug = str(slug or "").strip().lower()
        runtime = self._runtimes.pop(normalized_slug, None)
        if runtime is not None and runtime.module.__name__ in sys.modules:
            del sys.modules[runtime.module.__name__]

    def get_runtime(self, slug: str) -> DataSourceRuntime | None:
        normalized_slug = str(slug or "").strip().lower()
        return self._runtimes.get(normalized_slug)

    def runtime_snapshot(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for runtime in sorted(self._runtimes.values(), key=lambda item: item.slug):
            rows.append(
                {
                    "slug": runtime.slug,
                    "class_name": runtime.class_name,
                    "name": runtime.name,
                    "description": runtime.description,
                    "loaded_at": runtime.loaded_at.isoformat(),
                    "source_hash": runtime.source_hash,
                    "run_count": runtime.run_count,
                    "error_count": runtime.error_count,
                    "last_run": runtime.last_run.isoformat() if runtime.last_run else None,
                    "last_error": runtime.last_error,
                }
            )
        return rows

    async def refresh_all_from_db(self, session: AsyncSession | None = None) -> dict[str, Any]:
        from models.database import AsyncSessionLocal, DataSource

        own_session = False
        db: AsyncSession
        if session is None:
            db = AsyncSessionLocal()
            own_session = True
        else:
            db = session

        loaded: list[str] = []
        errors: dict[str, str] = {}
        db_state_changed = False

        try:
            rows = (
                (
                    await db.execute(
                        select(DataSource).order_by(DataSource.sort_order.asc(), DataSource.slug.asc())
                    )
                )
                .scalars()
                .all()
            )
            active_slugs: set[str] = set()

            for row in rows:
                slug = str(row.slug or "").strip().lower()
                if not slug:
                    continue
                active_slugs.add(slug)
                if not bool(row.enabled):
                    self.unload(slug)
                    row_changed = False
                    if row.status != "unloaded":
                        row.status = "unloaded"
                        row_changed = True
                    if row.error_message is not None:
                        row.error_message = None
                        row_changed = True
                    if row_changed:
                        db_state_changed = True
                    continue

                try:
                    runtime = self.load(
                        slug=slug,
                        source_code=str(row.source_code or ""),
                        config=dict(row.config or {}),
                        class_name=row.class_name,
                    )
                    row_changed = False
                    if row.status != "loaded":
                        row.status = "loaded"
                        row_changed = True
                    if row.error_message is not None:
                        row.error_message = None
                        row_changed = True
                    if row.class_name != runtime.class_name:
                        row.class_name = runtime.class_name
                        row_changed = True
                    if row_changed:
                        db_state_changed = True
                    loaded.append(slug)
                except DataSourceValidationError as exc:
                    error_text = str(exc)
                    row_changed = False
                    if row.status != "error":
                        row.status = "error"
                        row_changed = True
                    if row.error_message != error_text:
                        row.error_message = error_text
                        row_changed = True
                    if row_changed:
                        db_state_changed = True
                    errors[slug] = error_text
                except Exception as exc:  # pragma: no cover - safety net
                    error_text = str(exc)
                    row_changed = False
                    if row.status != "error":
                        row.status = "error"
                        row_changed = True
                    if row.error_message != error_text:
                        row.error_message = error_text
                        row_changed = True
                    if row_changed:
                        db_state_changed = True
                    errors[slug] = error_text

            for loaded_slug in list(self._runtimes.keys()):
                if loaded_slug not in active_slugs:
                    self.unload(loaded_slug)

            if db_state_changed:
                await db.commit()
        finally:
            if own_session:
                await db.close()

        return {"loaded": loaded, "errors": errors}


data_source_loader = DataSourceLoader()


__all__ = [
    "DATA_SOURCE_TEMPLATE",
    "DataSourceLoader",
    "DataSourceRuntime",
    "DataSourceValidationError",
    "data_source_loader",
    "validate_data_source_source",
]
