"""CI guard: forbid hardcoded magic constants in the fill simulator.

Every parameter that affects fill probability must be either:
  * a venue-defined constant (fees, tick size — hardcoded is fine);
  * a unit conversion;
  * read from ``services.fill_simulator.empirical_constants`` (learned
    from book_delta_events) or ``services.fill_simulator.latency``
    (measured from execution_latency_metrics) or operator override
    (set via the UI).

This script scans the two simulator-critical modules for raw float
literals that look like fill-model magic numbers (0.x, where x is
not 0 or 1) and fails the build if any new ones are introduced
without an explicit allowlist comment.

Usage:
    python scripts/ci/check_no_hardcoded_fill_constants.py
Exit code 0 = pass; non-zero = forbidden literal found.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[2] / "backend"

# Files that own the simulator math.  Anything else can have float
# literals freely; these two are the high-leverage spots where a
# hardcoded constant silently breaks calibration.
GUARDED_FILES = (
    BACKEND_ROOT / "services" / "simulation" / "fill_models.py",
    BACKEND_ROOT / "services" / "optimization" / "execution_estimator.py",
)

# Float literals we explicitly allow.  Each entry is (value, reason).
# The first match by exact equality wins.  Edit this list rather than
# sprinkling ``# noqa: fill-const`` comments through the code.
ALLOWED = {
    0.0,    # zero
    1.0,    # one / multiplicative identity
    -1.0,   # negative identity
    0.5,    # midpoint / half — frequently a unit fraction
    2.0,    # squaring / doubling
    100.0,  # percent denom
    10000.0,  # bps denom
    10_000.0,
    60.0,   # seconds-per-minute
    1000.0,  # ms-per-second / unit conversion
    0.0001,  # min price floor (Polymarket tick)
    0.9999,  # max price ceiling
    0.999999,  # epsilon-tight ceiling for log-prob clamps
    1e-09,  # divide-by-zero guard
    # Established baseline values that already lived in the simulator
    # math BEFORE the empirical_constants refactor.  New hardcoded
    # values are forbidden; these are grandfathered so the guard
    # ratchets new additions without forcing a full sweep on day one.
    # If you add to this list, document why in the commit message.
    0.20,   # fill_models.py participation cap (legacy)
    0.35,   # fill_models.py no-volume default fill ratio (legacy)
    0.88,   # fill_models.py displayed_depth_factor (legacy default; learned at runtime)
    0.65,   # execution_estimator.py maker_queue_ahead_fraction (legacy default)
    1.35,   # execution_estimator.py adjustment factor (legacy)
    10.0,   # execution_estimator.py constant
    20.0,   # execution_estimator.py constant
}

# Names whose RHS we ALWAYS ignore — these are the documented places
# where the "default" lives and gets overridden at call time by the
# empirical_constants / latency / cox modules.  In practice this is
# the ``ExecutionEstimatorConfig`` dataclass — keeping defaults there
# is fine because real callers override them; what we guard against
# is *new* magic numbers showing up in the math itself.
ALLOWED_TARGETS = {
    "fee_bps",
    "latency_ms",
    "time_in_force_seconds",
    "displayed_depth_factor",
    "min_depth_factor",
    "max_book_age_ms",
    "stale_depth_decay",
    "maker_queue_ahead_fraction",
    "maker_trade_flow_multiplier",
    "adverse_selection_multiplier",
    "recent_trade_lookback_seconds",
    "slippage_bps",
    "fee_bps",
    "latency_ms",
    "displayed_depth_factor",
    # FillConfig defaults — same rationale as ExecutionEstimatorConfig.
    # (FillConfig lives in fill_models.py.)
}


def _is_allowed_target(node: ast.AST) -> bool:
    parent = getattr(node, "_parent", None)
    if parent is None:
        return False
    if isinstance(parent, ast.AnnAssign) and isinstance(parent.target, ast.Name):
        return parent.target.id in ALLOWED_TARGETS
    if isinstance(parent, ast.Assign):
        for tgt in parent.targets:
            if isinstance(tgt, ast.Name) and tgt.id in ALLOWED_TARGETS:
                return True
    return False


def _walk_and_link_parents(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child._parent = node


def scan(path: Path) -> list[tuple[int, float]]:
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(path))
    _walk_and_link_parents(tree)
    offenders: list[tuple[int, float]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, float):
            value = float(node.value)
            if value in ALLOWED:
                continue
            if _is_allowed_target(node):
                continue
            # Allow integers used in arithmetic — they parse as int constants,
            # which we don't visit here.  Only raw floats trip the guard.
            offenders.append((node.lineno, value))
    return offenders


def main() -> int:
    failures: list[tuple[Path, int, float]] = []
    for path in GUARDED_FILES:
        if not path.exists():
            print(f"[skip] {path} not found", file=sys.stderr)
            continue
        for line, value in scan(path):
            failures.append((path, line, value))

    if not failures:
        print("ok: no hardcoded fill-model magic numbers detected")
        return 0

    print("FAIL: hardcoded fill-model magic numbers detected")
    print("Each of these must be either added to the ALLOWED set in")
    print("  scripts/ci/check_no_hardcoded_fill_constants.py")
    print("or moved into services/fill_simulator/empirical_constants.py")
    print("(if learned from data) / services/fill_simulator/latency.py")
    print("(if measured) / a UI-overridable config.")
    print()
    for path, line, value in failures:
        rel = path.relative_to(BACKEND_ROOT.parent)
        print(f"  {rel}:{line}  literal {value!r}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
