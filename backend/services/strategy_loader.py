"""
Unified Strategy Loader

The ONE loader for all strategies. Handles detect, evaluate, and exit phases.

Each strategy is a Python file defining a class that extends ``BaseStrategy``
and implements one or more of:

  - **detect()** / **detect_async()** — opportunity detection (scanner phase).
  - **evaluate(signal, context)** — execution gating (orchestrator phase).
  - **should_exit(position, market_state)** — exit logic (lifecycle phase).

The loader validates source via AST, enforces an import allow-list, compiles
in an isolated module, and exposes runtime status tracking for dashboards.
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import sys
import traceback
import types
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

from utils.logger import get_logger
from services.event_dispatcher import event_dispatcher

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Strategy template — shown to users as a starting point
# ---------------------------------------------------------------------------

STRATEGY_TEMPLATE = '''"""
Strategy: My Custom Strategy

A unified strategy that handles detection, execution gating, and exit logic.
Implement detect() to find opportunities, evaluate() to gate execution,
and should_exit() to manage open positions.
"""

from models import Market, Event, ArbitrageOpportunity
from services.strategies.base import BaseStrategy, StrategyDecision, ExitDecision, DecisionCheck


class MyCustomStrategy(BaseStrategy):
    """
    A custom unified strategy.

    Required: detect() or detect_async() — find opportunities
    Optional: evaluate() — custom execution gating (default: passthrough)
    Optional: should_exit() — custom exit logic (default: TP/SL/trailing)
    """

    name = "My Custom Strategy"
    description = "Describe what this strategy detects and how it trades"

    # Strategy metadata
    mispricing_type = "within_market"  # or "cross_market", "settlement_lag", "news_information"
    source_key = "scanner"
    worker_affinity = "scanner"
    opportunity_ttl_minutes = 45  # None = use global default
    allow_deduplication = True
    binary_only = True

    default_config = {
        # Detection params
        "example_threshold": 0.05,
        # Execution params
        "min_edge_percent": 3.0,
        "min_confidence": 0.4,
        "base_size_usd": 25.0,
        "max_size_usd": 200.0,
        # Exit params
        "take_profit_pct": 15.0,
        "stop_loss_pct": 8.0,
        "max_hold_minutes": None,
        "trailing_stop_pct": None,
    }

    def detect(self, events, markets, prices):
        """Find opportunities. Runs every scan cycle.

        Use self.state to persist data across cycles:
            prior = self.state.get("last_prices", {})
            self.state["last_prices"] = {m.id: m.yes_price for m in markets}
        """
        opportunities = []
        for market in markets:
            if market.closed or not market.active:
                continue
            # Your detection logic here...
            # opp = self.create_opportunity(...)
            # if opp:
            #     opp.strategy_context = {"my_signal_data": ...}
            #     opportunities.append(opp)
        return opportunities

    def evaluate(self, signal, context):
        """Gate execution. Called when orchestrator picks up signal."""
        params = context.get("params") or {}
        edge = float(getattr(signal, "edge_percent", 0) or 0)

        if edge < float(params.get("min_edge_percent", 3.0)):
            return StrategyDecision("skipped", f"Edge {edge:.1f}% too low")

        return StrategyDecision(
            "selected", "Signal approved",
            size_usd=float(params.get("base_size_usd", 25.0)),
            checks=[DecisionCheck("edge", "Edge check", True, score=edge)]
        )

    def should_exit(self, position, market_state):
        """Manage open positions. Called every lifecycle cycle."""
        # Fall back to standard TP/SL/trailing from config
        return self.default_exit_check(position, market_state)
'''


# ---------------------------------------------------------------------------
# Allowed imports (merged superset from plugin_loader + strategy_db_loader)
# ---------------------------------------------------------------------------

ALLOWED_IMPORT_PREFIXES = {
    # Future annotations
    "__future__",
    # App modules
    "models",
    "services.strategies",
    "services.strategies.base",
    "services.trader_orchestrator.strategies.base",
    "services.trader_orchestrator.strategies",
    "services.news",
    "services.optimization",
    "services.ws_feeds",
    "services.chainlink_feed",
    "services.fee_model",
    "services.ai",
    "services.strategy_sdk",
    "services.data_source_sdk",
    "services.weather",
    "services.data_events",
    "services.event_dispatcher",
    "config",
    "utils",
    # Standard library (safe subset)
    "math",
    "statistics",
    "collections",
    "datetime",
    "re",
    "json",
    "random",
    "threading",
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
    # Third party
    "httpx",
    "numpy",
    "scipy",
}

# Explicitly blocked imports (dangerous even if they happen to pass prefix)
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
    "urllib",  # urllib itself is blocked; urllib.parse is allowed via prefix
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

# Blocked bare function calls
_BLOCKED_CALL_NAMES = {"exec", "eval", "compile", "__import__", "open", "input"}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class StrategyValidationError(Exception):
    """Raised when strategy source code fails validation or loading."""

    pass


# ---------------------------------------------------------------------------
# AST validation helpers
# ---------------------------------------------------------------------------


def _check_imports(tree: ast.AST) -> list[str]:
    """Check all imports in the AST against the allow-list. Returns violations."""
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod_root = alias.name.split(".")[0]
                # Allow urllib.parse explicitly even though urllib is blocked
                if alias.name.startswith("urllib.parse"):
                    continue
                if mod_root in BLOCKED_IMPORTS:
                    violations.append(f"Blocked import: '{alias.name}' (line {node.lineno})")
                elif not any(alias.name.startswith(p) for p in ALLOWED_IMPORT_PREFIXES):
                    violations.append(
                        f"Disallowed import: '{alias.name}' (line {node.lineno}). "
                        f"Use standard library or app modules only."
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                mod_root = node.module.split(".")[0]
                if node.module.startswith("urllib.parse"):
                    continue
                if mod_root in BLOCKED_IMPORTS:
                    violations.append(f"Blocked import: 'from {node.module}' (line {node.lineno})")
                elif not any(node.module.startswith(p) for p in ALLOWED_IMPORT_PREFIXES):
                    violations.append(
                        f"Disallowed import: 'from {node.module}' (line {node.lineno}). "
                        f"Use standard library or app modules only."
                    )
    return violations


def _check_blocked_calls(tree: ast.AST) -> list[str]:
    """Check for dangerous bare function calls. Returns violations."""
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


def _find_strategy_class(tree: ast.AST, class_name: Optional[str] = None) -> Optional[str]:
    """Find a class extending BaseStrategy in the AST.

    If *class_name* is given, look for that exact class; otherwise pick the
    first class whose name ends with ``Strategy`` or fallback to the first
    class that extends ``BaseStrategy``.
    """
    base_names = {"BaseStrategy"}

    # If explicit class_name requested, look for it
    if class_name:
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                return class_name
        return None

    # Auto-detect: first class extending BaseStrategy
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                base_id = None
                if isinstance(base, ast.Name):
                    base_id = base.id
                elif isinstance(base, ast.Attribute):
                    base_id = base.attr
                if base_id in base_names:
                    return node.name

    # Fallback: first class whose name contains "Strategy"
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and "Strategy" in node.name:
            return node.name

    return None


def _extract_class_capabilities(tree: ast.AST, class_name: str) -> dict:
    """Return which methods are present on the strategy class."""
    result = {
        "has_detect": False,
        "has_detect_async": False,
        "has_on_event": False,
        "has_evaluate": False,
        "has_should_exit": False,
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if item.name == "detect":
                        result["has_detect"] = True
                    elif item.name == "detect_async":
                        result["has_detect_async"] = True
                    elif item.name == "on_event":
                        result["has_on_event"] = True
                    elif item.name == "evaluate":
                        result["has_evaluate"] = True
                    elif item.name == "should_exit":
                        result["has_should_exit"] = True
            break
    return result


def _extract_class_attribute(tree: ast.AST, class_name: str, attr_name: str) -> Optional[str]:
    """Extract a string class attribute value from the AST."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.Assign):
                    for target in item.targets:
                        if isinstance(target, ast.Name) and target.id == attr_name:
                            if isinstance(item.value, ast.Constant):
                                return str(item.value.value)
    return None


# ---------------------------------------------------------------------------
# Public validation function
# ---------------------------------------------------------------------------


def validate_strategy_source(
    source_code: str,
    class_name: Optional[str] = None,
) -> dict:
    """Validate strategy source code without executing it.

    Returns a dict with:
        valid (bool): Whether the code is valid
        class_name (str|None): Name of the strategy class found
        strategy_name (str|None): Value of the 'name' attribute
        strategy_description (str|None): Value of the 'description' attribute
        capabilities (dict): Which methods are defined
        errors (list[str]): Validation error messages
        warnings (list[str]): Non-fatal warnings
    """
    result: dict[str, Any] = {
        "valid": False,
        "class_name": None,
        "strategy_name": None,
        "strategy_description": None,
        "capabilities": None,
        "errors": [],
        "warnings": [],
    }

    # 1. Syntax check
    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        result["errors"].append(f"Syntax error at line {e.lineno}: {e.msg}")
        return result

    # 2. Import safety check
    import_violations = _check_imports(tree)
    if import_violations:
        result["errors"].extend(import_violations)
        return result

    # 3. Blocked call check
    call_violations = _check_blocked_calls(tree)
    if call_violations:
        result["errors"].extend(call_violations)
        return result

    # 4. Find strategy class
    found_class = _find_strategy_class(tree, class_name)
    if not found_class:
        if class_name:
            result["errors"].append(f"Class '{class_name}' was not found in strategy source.")
        else:
            result["errors"].append(
                "No class extending BaseStrategy found. "
                "Your strategy must define a class like: class MyStrategy(BaseStrategy):"
            )
        return result
    result["class_name"] = found_class

    # 5. Check capabilities
    capabilities = _extract_class_capabilities(tree, found_class)
    result["capabilities"] = capabilities

    # 6. Require at least one of detect, detect_async, on_event, or evaluate.
    # - detect() / detect_async(): scanner strategies (base on_event routes to them)
    # - on_event(): event-driven strategies reacting to non-scanner data
    # - evaluate(): execution-only strategies (signals generated externally)
    has_any = (
        capabilities["has_detect"]
        or capabilities["has_detect_async"]
        or capabilities["has_on_event"]
        or capabilities["has_evaluate"]
    )
    if not has_any:
        result["errors"].append(
            f"Class '{found_class}' must implement at least one of: detect(), "
            f"detect_async(), on_event(), or evaluate(). "
            f"Use detect() for synchronous scanner strategies, detect_async() for "
            f"async I/O-bound strategies, on_event() for event-driven strategies "
            f"reacting to crypto/weather/news data, or evaluate() for "
            f"execution-gating of externally generated signals."
        )
        return result

    # 7. Extract metadata
    strategy_name = _extract_class_attribute(tree, found_class, "name")
    strategy_description = _extract_class_attribute(tree, found_class, "description")

    if not strategy_name:
        result["warnings"].append(f"Class '{found_class}' has no 'name' attribute. A default name will be used.")
    if not strategy_description:
        result["warnings"].append(
            f"Class '{found_class}' has no 'description' attribute. A default description will be used."
        )

    result["strategy_name"] = strategy_name
    result["strategy_description"] = strategy_description
    result["valid"] = True
    return result


# ---------------------------------------------------------------------------
# Runtime state for a loaded strategy
# ---------------------------------------------------------------------------


@dataclass
class StrategyAvailability:
    """Whether a strategy is available for use."""

    available: bool
    strategy_key: str
    resolved_key: str
    reason: Optional[str] = None


@dataclass
class LoadedStrategy:
    """Runtime state for a loaded strategy."""

    slug: str
    instance: Any  # BaseStrategy instance
    class_name: str
    source_hash: str
    loaded_at: datetime
    run_count: int = 0
    error_count: int = 0
    total_opportunities: int = 0
    last_run: Optional[datetime] = None
    last_error: Optional[str] = None


def _strategy_runtime_hash(source_code: str, config: Optional[dict]) -> str:
    """Hash strategy source + config so config-only changes trigger reloads."""
    normalized_config = config if isinstance(config, dict) else {}
    serialized = json.dumps(normalized_config, sort_keys=True, separators=(",", ":"))
    payload = f"{source_code}\n{serialized}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Strategy Loader
# ---------------------------------------------------------------------------


class StrategyLoader:
    """Manages loading, validation, and lifecycle of strategies from source code.

    This is the SINGLE loader for all strategies — detection, evaluation, and exit.
    """

    def __init__(self) -> None:
        self._loaded: dict[str, LoadedStrategy] = {}
        self._errors: dict[str, str] = {}
        self._module_counter: int = 0
        self._refresh_lock: Optional[asyncio.Lock] = None

    @property
    def refresh_lock(self) -> asyncio.Lock:
        if self._refresh_lock is None:
            self._refresh_lock = asyncio.Lock()
        return self._refresh_lock

    # ── Load / Unload ────────────────────────────────────────

    def load(
        self,
        slug: str,
        source_code: str,
        config: Optional[dict] = None,
    ) -> LoadedStrategy:
        """Load a strategy from source code.

        Validates the source via AST, compiles in an isolated module,
        instantiates the strategy class, and stores it in ``_loaded``.

        Args:
            slug: Unique identifier for this strategy.
            source_code: Python source code defining a BaseStrategy subclass.
            config: Optional config overrides passed to ``instance.configure()``.

        Returns:
            LoadedStrategy instance.

        Raises:
            StrategyValidationError: If the source code is invalid or loading fails.
        """
        # Validate
        validation = validate_strategy_source(source_code)
        if not validation["valid"]:
            raise StrategyValidationError("Strategy validation failed:\n" + "\n".join(validation["errors"]))

        class_name = validation["class_name"]

        # Unload existing version if present
        if slug in self._loaded:
            self.unload(slug)

        # Create a unique module name
        self._module_counter += 1
        module_name = f"_strategy_{slug}_{self._module_counter}"

        try:
            # Compile
            code = compile(source_code, f"<strategy:{slug}>", "exec")

            # Create isolated module
            module = types.ModuleType(module_name)
            module.__file__ = f"<strategy:{slug}>"
            module.__builtins__ = __builtins__

            # Register so imports within the module resolve
            sys.modules[module_name] = module

            # Execute in the module's namespace
            exec(code, module.__dict__)  # noqa: S102

            # Retrieve the strategy class
            strategy_class = getattr(module, class_name, None)
            if strategy_class is None:
                raise StrategyValidationError(
                    f"Class '{class_name}' not found after loading. This is likely a bug in the strategy loader."
                )

            # Verify it extends BaseStrategy
            from services.strategies.base import BaseStrategy

            is_valid_class = isinstance(strategy_class, type) and issubclass(strategy_class, BaseStrategy)
            if not is_valid_class:
                raise StrategyValidationError(f"Class '{class_name}' does not extend BaseStrategy.")

            # Set strategy_type to slug
            strategy_class.strategy_type = slug

            # Default name / description if missing
            if not getattr(strategy_class, "name", None):
                strategy_class.name = slug.replace("_", " ").title()
            if not getattr(strategy_class, "description", None):
                strategy_class.description = f"Strategy: {slug}"

            # Instantiate
            instance = strategy_class()

            # Set key for orchestrator lookups
            instance.key = slug

            # Apply config
            instance.configure(config or {})

            # Source hash for change detection
            source_hash = _strategy_runtime_hash(source_code, config or {})

            loaded = LoadedStrategy(
                slug=slug,
                instance=instance,
                class_name=class_name,
                source_hash=source_hash,
                loaded_at=datetime.now(timezone.utc),
            )
            self._loaded[slug] = loaded

            # Register event subscriptions
            subs = getattr(instance, "subscriptions", [])
            if subs:
                event_dispatcher.unsubscribe_all(slug)  # Clear stale subscriptions
                for event_type in subs:
                    event_dispatcher.subscribe(slug, event_type, instance.on_event)

            caps = validation.get("capabilities") or {}
            cap_parts = [k.replace("has_", "") for k, v in caps.items() if v]
            logger.info(
                "Strategy loaded: %s (class=%s, name='%s', capabilities=[%s])",
                slug,
                class_name,
                instance.name,
                ", ".join(cap_parts),
            )
            return loaded

        except StrategyValidationError:
            sys.modules.pop(module_name, None)
            raise
        except Exception as e:
            sys.modules.pop(module_name, None)
            tb = traceback.format_exc()
            raise StrategyValidationError(f"Failed to load strategy '{slug}': {e}\n\n{tb}") from e

    def unload(self, slug: str) -> None:
        """Unload a strategy by slug."""
        event_dispatcher.unsubscribe_all(slug)
        loaded = self._loaded.pop(slug, None)
        if loaded:
            # Clean up module from sys.modules
            for name in list(sys.modules.keys()):
                if name.startswith(f"_strategy_{slug}_"):
                    sys.modules.pop(name, None)
            logger.info("Strategy unloaded: %s", slug)

    # ── DB reload ────────────────────────────────────────────

    async def reload_from_db(
        self,
        slug: str,
        session: "AsyncSession",
    ) -> dict:
        """Reload a single strategy from the ``strategies`` DB table.

        Reads the ``Strategy`` row by slug, validates, compiles, and loads.
        Updates the row's ``status`` and ``error_message`` columns.

        Args:
            slug: Strategy slug to reload.
            session: SQLAlchemy async session (caller manages commit/close).

        Returns:
            Dict with ``"status"`` key: ``"loaded"``, ``"not_found"``,
            ``"disabled"``, or ``"error"``.
        """
        from models.database import Strategy
        from sqlalchemy import select

        row = (await session.execute(select(Strategy).where(Strategy.slug == slug))).scalar_one_or_none()

        if not row:
            return {"status": "not_found", "slug": slug}

        if not row.enabled:
            # Unload if previously loaded
            self.unload(slug)
            row.status = "disabled"
            row.error_message = None
            return {"status": "disabled", "slug": slug}

        try:
            config = row.config if isinstance(row.config, dict) else {}
            loaded = self.load(
                slug=slug,
                source_code=row.source_code or "",
                config=config,
            )
            row.status = "loaded"
            row.error_message = None
            return {
                "status": "loaded",
                "slug": slug,
                "class_name": loaded.class_name,
                "source_hash": loaded.source_hash,
            }
        except Exception as exc:
            tb = traceback.format_exc()
            error_msg = str(exc)
            row.status = "error"
            row.error_message = f"{error_msg}\n{tb}"
            logger.error("Failed to reload strategy '%s': %s", slug, error_msg)
            return {
                "status": "error",
                "slug": slug,
                "error": error_msg,
            }

    async def refresh_all_from_db(
        self,
        *,
        session: "AsyncSession",
        strategy_keys: Optional[list[str]] = None,
        source_keys: Optional[list[str]] = None,
        prune_unlisted: bool = False,
    ) -> dict[str, Any]:
        """Bulk-load all enabled strategies from the DB.

        Replaces the in-memory cache with fresh data. Disabled strategies
        are unloaded. Rows are updated with ``status`` and ``error_message``.

        Args:
            session: SQLAlchemy async session (caller manages commit/close).
            strategy_keys: Optional subset of slugs to refresh. If None,
                refreshes all rows.
            source_keys: Optional subset of source keys to refresh.
            prune_unlisted: If True, unload loaded strategies not present in
                the selected result set. Full refreshes always prune deletions.

        Returns:
            Dict with ``"loaded"``, ``"errors"`` keys.
        """
        from models.database import Strategy
        from sqlalchemy import func, select
        from utils.utcnow import utcnow

        keys_filter = {str(key or "").strip().lower() for key in (strategy_keys or []) if str(key or "").strip()}
        source_filter = {str(key or "").strip().lower() for key in (source_keys or []) if str(key or "").strip()}

        async with self.refresh_lock:
            query = select(Strategy).order_by(Strategy.slug.asc())
            if keys_filter:
                query = query.where(func.lower(func.coalesce(Strategy.slug, "")).in_(tuple(sorted(keys_filter))))
            if source_filter:
                query = query.where(
                    func.lower(func.coalesce(Strategy.source_key, "")).in_(tuple(sorted(source_filter)))
                )
            rows = list((await session.execute(query)).scalars().all())

            next_loaded = dict(self._loaded)
            next_errors = dict(self._errors)
            db_state_changed = False
            selected_slugs = {str(row.slug or "").strip().lower() for row in rows if str(row.slug or "").strip()}

            is_filtered_refresh = bool(keys_filter or source_filter)
            should_prune = bool(prune_unlisted or not is_filtered_refresh)
            if should_prune:
                for loaded_slug in list(next_loaded.keys()):
                    if loaded_slug in selected_slugs:
                        continue
                    self.unload(loaded_slug)
                    next_loaded.pop(loaded_slug, None)
                    next_errors.pop(loaded_slug, None)

            for row in rows:
                slug = str(row.slug or "").strip().lower()
                row_config = row.config if isinstance(row.config, dict) else {}

                # Compute hash of the incoming source code for change detection
                new_source_hash = _strategy_runtime_hash(row.source_code or "", row_config)
                current_loaded = next_loaded.get(slug)

                # Skip reload if already loaded with identical source and still enabled
                if current_loaded and current_loaded.source_hash == new_source_hash and bool(row.enabled):
                    row_changed = False
                    if row.status != "loaded":
                        row.status = "loaded"
                        row_changed = True
                    if row.error_message is not None:
                        row.error_message = None
                        row_changed = True
                    if row_changed:
                        row.updated_at = utcnow()
                        db_state_changed = True
                    continue

                # Clear previous state for this slug
                if slug in next_loaded:
                    self.unload(slug)
                    next_loaded.pop(slug, None)
                next_errors.pop(slug, None)

                if not bool(row.enabled):
                    row_changed = False
                    if row.status != "disabled":
                        row.status = "disabled"
                        row_changed = True
                    if row.error_message is not None:
                        row.error_message = None
                        row_changed = True
                    if row_changed:
                        row.updated_at = utcnow()
                        db_state_changed = True
                    continue

                try:
                    loaded = self.load(
                        slug=slug,
                        source_code=row.source_code or "",
                        config=row_config,
                    )
                    next_loaded[slug] = loaded
                    row_changed = False
                    if row.status != "loaded":
                        row.status = "loaded"
                        row_changed = True
                    if row.error_message is not None:
                        row.error_message = None
                        row_changed = True
                    if row_changed:
                        row.updated_at = utcnow()
                        db_state_changed = True
                except Exception as exc:
                    error_message = str(exc)
                    tb = traceback.format_exc()
                    next_errors[slug] = error_message
                    formatted_error = f"{error_message}\n{tb}"
                    row_changed = False
                    if row.status != "error":
                        row.status = "error"
                        row_changed = True
                    if row.error_message != formatted_error:
                        row.error_message = formatted_error
                        row_changed = True
                    if row_changed:
                        row.updated_at = utcnow()
                        db_state_changed = True

            self._loaded = next_loaded
            self._errors = next_errors

            if db_state_changed:
                await session.commit()

            return {
                "loaded": sorted(self._loaded.keys()),
                "errors": dict(self._errors),
            }

    async def refresh_from_db(
        self,
        *,
        session: "AsyncSession",
        strategy_keys: Optional[list[str]] = None,
        source_keys: Optional[list[str]] = None,
        prune_unlisted: bool = False,
    ) -> dict[str, Any]:
        return await self.refresh_all_from_db(
            session=session,
            strategy_keys=strategy_keys,
            source_keys=source_keys,
            prune_unlisted=prune_unlisted,
        )

    # ── Accessors ────────────────────────────────────────────

    def get_strategy(self, slug: str) -> Optional[LoadedStrategy]:
        """Return the LoadedStrategy for *slug*, or None."""
        return self._loaded.get(slug)

    def get_instance(self, slug: str) -> Any:
        """Return the BaseStrategy instance for *slug*, or None."""
        loaded = self._loaded.get(slug)
        return loaded.instance if loaded else None

    def get_all_instances(self) -> list:
        """Return all loaded strategy instances (e.g. for the scanner)."""
        return [ls.instance for ls in self._loaded.values()]

    def is_loaded(self, slug: str) -> bool:
        """Check whether a strategy is currently loaded."""
        return slug in self._loaded

    def get_availability(self, strategy_key: str) -> StrategyAvailability:
        """Check if a strategy is loaded and available for use."""
        slug = str(strategy_key or "").strip().lower()
        if slug in self._loaded:
            return StrategyAvailability(available=True, strategy_key=strategy_key, resolved_key=slug)
        reason = self._errors.get(slug)
        return StrategyAvailability(
            available=False,
            strategy_key=strategy_key,
            resolved_key=slug,
            reason=reason or f"strategy_unavailable:{slug}",
        )

    def get_runtime_status(self, slug: str) -> Optional[dict]:
        """Return full runtime status for a loaded strategy (used by serializers)."""
        loaded = self._loaded.get(slug)
        if loaded is not None:
            return {
                "strategy_key": loaded.slug,
                "class_name": loaded.class_name,
                "source_hash": loaded.source_hash,
                "status": "loaded",
                "error_message": None,
                "loaded_at": loaded.loaded_at.isoformat(),
            }
        if slug in self._errors:
            return {
                "strategy_key": slug,
                "status": "error",
                "error_message": self._errors[slug],
            }
        return None

    # ── Run tracking ─────────────────────────────────────────

    def record_run(
        self,
        slug: str,
        opportunities_found: int = 0,
        error: Optional[str] = None,
    ) -> None:
        """Record a run result for status tracking."""
        loaded = self._loaded.get(slug)
        if not loaded:
            return
        loaded.run_count += 1
        loaded.last_run = datetime.now(timezone.utc)
        if error:
            loaded.error_count += 1
            loaded.last_error = error
        else:
            loaded.total_opportunities += opportunities_found

    # ── Status / introspection ───────────────────────────────

    def get_status(self, slug: str) -> Optional[dict]:
        """Get status info for a loaded strategy."""
        loaded = self._loaded.get(slug)
        if not loaded:
            return None
        return {
            "slug": loaded.slug,
            "class_name": loaded.class_name,
            "name": getattr(loaded.instance, "name", loaded.slug),
            "description": getattr(loaded.instance, "description", ""),
            "source_hash": loaded.source_hash,
            "loaded_at": loaded.loaded_at.isoformat(),
            "run_count": loaded.run_count,
            "error_count": loaded.error_count,
            "total_opportunities": loaded.total_opportunities,
            "last_run": loaded.last_run.isoformat() if loaded.last_run else None,
            "last_error": loaded.last_error,
        }

    def get_all_statuses(self) -> list[dict]:
        """Get status for all loaded strategies."""
        return [self.get_status(slug) for slug in self._loaded if self.get_status(slug) is not None]


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

strategy_loader = StrategyLoader()


# ---------------------------------------------------------------------------
# Serialization helper (replaces strategy_db_loader.serialize_strategy_definition)
# ---------------------------------------------------------------------------


def serialize_strategy_definition(row: Any) -> dict[str, Any]:
    """Serialize a Strategy ORM row to a dict for API responses."""
    runtime = strategy_loader.get_runtime_status(str(row.slug or ""))
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
        "aliases_json": [],  # Deprecated — aliases system removed
        "is_system": bool(row.is_system),
        "enabled": bool(row.enabled),
        "status": row.status,
        "error_message": row.error_message,
        "version": int(row.version or 1),
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        "runtime": runtime,
    }
