"""
Plugin Loader

Dynamically loads user-written strategy plugins from source code stored in
the database. Each plugin is a Python file that defines a class extending
BaseStrategy with a detect() method — exactly like the built-in strategies.

The loader validates, compiles, and instantiates plugins at runtime, making
them available to the scanner alongside the built-in strategies.
"""

import ast
import sys
import types
import traceback
from datetime import datetime, timezone
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Plugin template — shown to users as a starting point
# ---------------------------------------------------------------------------

PLUGIN_TEMPLATE = '''"""
Plugin: My Custom Strategy

Describe what your strategy does here. This docstring is for your reference.
"""

from models import Market, Event, ArbitrageOpportunity
from services.strategies.base import BaseStrategy


class MyCustomStrategy(BaseStrategy):
    """
    A custom arbitrage detection strategy.

    Attributes:
        strategy_type: Unique slug for this plugin (set automatically by the loader).
        name: Human-readable name shown in the UI.
        description: Short description shown in the strategy list.
    """

    name = "My Custom Strategy"
    description = "Describe what this strategy detects"

    # Optional: default config values (users can override in the UI)
    default_config = {
        "example_threshold": 0.05,
    }

    def detect(
        self,
        events: list[Event],
        markets: list[Market],
        prices: dict[str, dict],
    ) -> list[ArbitrageOpportunity]:
        """
        Detect arbitrage opportunities.

        This method is called every scan cycle with the full set of active
        events, markets, and live CLOB prices.

        Args:
            events: All active Polymarket events.
            markets: All active markets across events.
            prices: Live CLOB mid-prices keyed by token ID.
                    Format: { token_id: { "mid": float, "best_bid": float, "best_ask": float } }

        Returns:
            List of detected ArbitrageOpportunity objects.

        Tips:
            - Use self.create_opportunity() to build opportunities (applies
              hard filters like min liquidity, min ROI, fee model, etc.)
            - Access config via self.config (dict with your default_config
              values, overridden by user settings)
            - Use self.fee for the current fee rate
            - Use self.min_profit for the minimum profit threshold
            - Filter out closed/inactive markets early for performance
            - Return an empty list if no opportunities are found
        """
        opportunities = []

        for market in markets:
            # Skip closed or inactive markets
            if market.closed or not market.active:
                continue

            # Skip non-binary markets
            if len(market.outcome_prices) != 2:
                continue

            # Get prices — use live CLOB prices when available
            yes_price = market.yes_price
            no_price = market.no_price

            if market.clob_token_ids:
                if len(market.clob_token_ids) > 0:
                    token = market.clob_token_ids[0]
                    if token in prices:
                        yes_price = prices[token].get("mid", yes_price)
                if len(market.clob_token_ids) > 1:
                    token = market.clob_token_ids[1]
                    if token in prices:
                        no_price = prices[token].get("mid", no_price)

            # ---- Your detection logic here ----
            # Example: find markets where YES + NO is unusually low
            total_cost = yes_price + no_price
            threshold = self.config.get("example_threshold", 0.05)

            if total_cost >= (1.0 - threshold):
                continue

            # Build positions
            positions = [
                {
                    "action": "BUY",
                    "outcome": "YES",
                    "price": yes_price,
                    "token_id": market.clob_token_ids[0] if market.clob_token_ids else None,
                },
                {
                    "action": "BUY",
                    "outcome": "NO",
                    "price": no_price,
                    "token_id": market.clob_token_ids[1]
                    if market.clob_token_ids and len(market.clob_token_ids) > 1
                    else None,
                },
            ]

            opp = self.create_opportunity(
                title=f"My Strategy: {market.question[:50]}",
                description=f"Custom detection on {market.question[:80]}",
                total_cost=total_cost,
                markets=[market],
                positions=positions,
            )
            if opp:
                opportunities.append(opp)

        return opportunities
'''


# ---------------------------------------------------------------------------
# Allowed imports for plugins (safety guardrails)
# ---------------------------------------------------------------------------

# Modules plugins are allowed to import. We allow the core app modules
# plus standard library math/data utilities.
ALLOWED_IMPORT_PREFIXES = {
    # App modules
    "models",
    "services.strategies.base",
    "services.fee_model",
    "config",
    # Standard library (safe subset)
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
    "time",
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
    # Data processing
    "numpy",
    "scipy",
}

# Explicitly blocked imports (dangerous even if they pass prefix check)
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
    "ast",
    "dis",
    "inspect",
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


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class PluginValidationError(Exception):
    """Raised when plugin source code fails validation."""

    pass


def _check_imports(tree: ast.AST) -> list[str]:
    """Check all imports in the AST for safety. Returns list of violations."""
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod = alias.name.split(".")[0]
                if mod in BLOCKED_IMPORTS:
                    violations.append(f"Blocked import: '{alias.name}' (line {node.lineno})")
                elif not any(alias.name.startswith(p) for p in ALLOWED_IMPORT_PREFIXES):
                    violations.append(
                        f"Disallowed import: '{alias.name}' (line {node.lineno}). "
                        f"Use standard library or app modules only."
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                mod = node.module.split(".")[0]
                if mod in BLOCKED_IMPORTS:
                    violations.append(f"Blocked import: 'from {node.module}' (line {node.lineno})")
                elif not any(node.module.startswith(p) for p in ALLOWED_IMPORT_PREFIXES):
                    violations.append(
                        f"Disallowed import: 'from {node.module}' (line {node.lineno}). "
                        f"Use standard library or app modules only."
                    )
    return violations


def _find_strategy_class(tree: ast.AST) -> Optional[str]:
    """Find the class that extends BaseStrategy in the AST. Returns class name or None."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                base_name = None
                if isinstance(base, ast.Name):
                    base_name = base.id
                elif isinstance(base, ast.Attribute):
                    base_name = base.attr
                if base_name == "BaseStrategy":
                    return node.name
    return None


def _check_detect_method(tree: ast.AST, class_name: str) -> bool:
    """Check that the strategy class has a detect method."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if item.name == "detect":
                        return True
    return False


def _check_name_attribute(tree: ast.AST, class_name: str) -> Optional[str]:
    """Extract the 'name' class attribute from the strategy class."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.Assign):
                    for target in item.targets:
                        if isinstance(target, ast.Name) and target.id == "name":
                            if isinstance(item.value, ast.Constant):
                                return str(item.value.value)
    return None


def _check_description_attribute(tree: ast.AST, class_name: str) -> Optional[str]:
    """Extract the 'description' class attribute from the strategy class."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.Assign):
                    for target in item.targets:
                        if isinstance(target, ast.Name) and target.id == "description":
                            if isinstance(item.value, ast.Constant):
                                return str(item.value.value)
    return None


def validate_plugin_source(source_code: str) -> dict:
    """Validate plugin source code without executing it.

    Returns a dict with:
        valid (bool): Whether the code is valid
        class_name (str|None): Name of the strategy class found
        strategy_name (str|None): Value of the 'name' attribute
        strategy_description (str|None): Value of the 'description' attribute
        errors (list[str]): List of validation error messages
        warnings (list[str]): List of non-fatal warnings
    """
    result = {
        "valid": False,
        "class_name": None,
        "strategy_name": None,
        "strategy_description": None,
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

    # 3. Find strategy class
    class_name = _find_strategy_class(tree)
    if not class_name:
        result["errors"].append(
            "No class extending BaseStrategy found. "
            "Your plugin must define a class like: class MyStrategy(BaseStrategy):"
        )
        return result
    result["class_name"] = class_name

    # 4. Check detect() method
    if not _check_detect_method(tree, class_name):
        result["errors"].append(
            f"Class '{class_name}' must implement a detect() method. "
            f"This method receives (events, markets, prices) and returns "
            f"a list of ArbitrageOpportunity objects."
        )
        return result

    # 5. Extract metadata
    name = _check_name_attribute(tree, class_name)
    description = _check_description_attribute(tree, class_name)

    if not name:
        result["warnings"].append(f"Class '{class_name}' has no 'name' attribute. A default name will be used.")
    if not description:
        result["warnings"].append(
            f"Class '{class_name}' has no 'description' attribute. A default description will be used."
        )

    result["strategy_name"] = name
    result["strategy_description"] = description
    result["valid"] = True
    return result


# ---------------------------------------------------------------------------
# Plugin loading
# ---------------------------------------------------------------------------


class LoadedPlugin:
    """A plugin that has been successfully loaded and instantiated."""

    def __init__(
        self,
        slug: str,
        instance: object,
        class_name: str,
        source_hash: str,
        loaded_at: datetime,
    ):
        self.slug = slug
        self.instance = instance  # BaseStrategy subclass instance
        self.class_name = class_name
        self.source_hash = source_hash
        self.loaded_at = loaded_at
        self.last_error: Optional[str] = None
        self.last_run: Optional[datetime] = None
        self.run_count: int = 0
        self.error_count: int = 0
        self.total_opportunities: int = 0


class PluginLoader:
    """Manages loading, validation, and lifecycle of strategy plugins."""

    def __init__(self):
        self._loaded: dict[str, LoadedPlugin] = {}  # slug -> LoadedPlugin
        self._module_counter = 0  # For unique module names

    @property
    def loaded_plugins(self) -> dict[str, LoadedPlugin]:
        return dict(self._loaded)

    def get_plugin(self, slug: str) -> Optional[LoadedPlugin]:
        return self._loaded.get(slug)

    def get_all_strategy_instances(self) -> list:
        """Return all loaded plugin strategy instances (for the scanner)."""
        return [p.instance for p in self._loaded.values()]

    def load_plugin(
        self,
        slug: str,
        source_code: str,
        config: Optional[dict] = None,
    ) -> LoadedPlugin:
        """Load a plugin from source code.

        Args:
            slug: Unique identifier for this plugin.
            source_code: Python source code defining a BaseStrategy subclass.
            config: Optional config overrides for the plugin.

        Returns:
            LoadedPlugin instance.

        Raises:
            PluginValidationError: If the source code is invalid.
        """
        # Validate first
        validation = validate_plugin_source(source_code)
        if not validation["valid"]:
            raise PluginValidationError("Plugin validation failed:\n" + "\n".join(validation["errors"]))

        class_name = validation["class_name"]

        # Unload existing version if any
        if slug in self._loaded:
            self.unload_plugin(slug)

        # Create a unique module name for this plugin
        self._module_counter += 1
        module_name = f"_plugin_{slug}_{self._module_counter}"

        try:
            # Compile the source code
            code = compile(source_code, f"<plugin:{slug}>", "exec")

            # Create a module for the plugin
            module = types.ModuleType(module_name)
            module.__file__ = f"<plugin:{slug}>"

            # Give the module access to builtins and the app's import system
            module.__builtins__ = __builtins__

            # Register the module so imports within it work
            sys.modules[module_name] = module

            # Execute the plugin code in the module's namespace
            exec(code, module.__dict__)  # noqa: S102

            # Find the strategy class
            strategy_class = getattr(module, class_name, None)
            if strategy_class is None:
                raise PluginValidationError(
                    f"Class '{class_name}' not found after loading. This is likely a bug in the plugin loader."
                )

            # Verify it's a subclass of BaseStrategy
            from services.strategies.base import BaseStrategy

            if not (isinstance(strategy_class, type) and issubclass(strategy_class, BaseStrategy)):
                raise PluginValidationError(f"Class '{class_name}' does not extend BaseStrategy.")

            # Set the strategy_type to the plugin's slug
            strategy_class.strategy_type = slug

            # Set name/description defaults if not provided
            if not hasattr(strategy_class, "name") or not strategy_class.name:
                strategy_class.name = slug.replace("_", " ").title()
            if not hasattr(strategy_class, "description") or not strategy_class.description:
                strategy_class.description = f"Plugin strategy: {slug}"

            # Instantiate the strategy
            instance = strategy_class()

            # Apply config overrides
            if config:
                # Merge with default_config if it exists
                default_config = getattr(instance, "default_config", {})
                merged_config = {**default_config, **config}
                instance.config = merged_config
            else:
                instance.config = getattr(instance, "default_config", {})

            # Calculate source hash for change detection
            import hashlib

            source_hash = hashlib.sha256(source_code.encode()).hexdigest()[:16]

            loaded = LoadedPlugin(
                slug=slug,
                instance=instance,
                class_name=class_name,
                source_hash=source_hash,
                loaded_at=datetime.now(timezone.utc),
            )
            self._loaded[slug] = loaded

            logger.info(f"Plugin loaded: {slug} (class={class_name}, name='{instance.name}')")
            return loaded

        except PluginValidationError:
            # Clean up the module on failure
            sys.modules.pop(module_name, None)
            raise
        except Exception as e:
            sys.modules.pop(module_name, None)
            tb = traceback.format_exc()
            raise PluginValidationError(f"Failed to load plugin '{slug}': {e}\n\n{tb}") from e

    def unload_plugin(self, slug: str) -> bool:
        """Unload a plugin by slug. Returns True if it was loaded."""
        loaded = self._loaded.pop(slug, None)
        if loaded:
            # Clean up the module from sys.modules
            module_name = None
            for name in list(sys.modules.keys()):
                if name.startswith(f"_plugin_{slug}_"):
                    module_name = name
                    break
            if module_name:
                sys.modules.pop(module_name, None)

            logger.info(f"Plugin unloaded: {slug}")
            return True
        return False

    def reload_plugin(
        self,
        slug: str,
        source_code: str,
        config: Optional[dict] = None,
    ) -> LoadedPlugin:
        """Reload a plugin (unload + load). Same as load_plugin but explicit."""
        return self.load_plugin(slug, source_code, config)

    def record_run(self, slug: str, opportunities_found: int, error: Optional[str] = None):
        """Record a plugin run result for status tracking."""
        plugin = self._loaded.get(slug)
        if not plugin:
            return
        plugin.run_count += 1
        plugin.last_run = datetime.now(timezone.utc)
        if error:
            plugin.error_count += 1
            plugin.last_error = error
        else:
            plugin.total_opportunities += opportunities_found

    def get_status(self, slug: str) -> Optional[dict]:
        """Get status info for a loaded plugin."""
        plugin = self._loaded.get(slug)
        if not plugin:
            return None
        return {
            "slug": plugin.slug,
            "class_name": plugin.class_name,
            "name": plugin.instance.name,
            "description": plugin.instance.description,
            "loaded_at": plugin.loaded_at.isoformat(),
            "source_hash": plugin.source_hash,
            "run_count": plugin.run_count,
            "error_count": plugin.error_count,
            "total_opportunities": plugin.total_opportunities,
            "last_run": plugin.last_run.isoformat() if plugin.last_run else None,
            "last_error": plugin.last_error,
        }

    def get_all_statuses(self) -> list[dict]:
        """Get status for all loaded plugins."""
        return [self.get_status(slug) for slug in self._loaded]


# Module-level singleton
plugin_loader = PluginLoader()
