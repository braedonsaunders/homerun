import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from main import app


def _all_route_paths(routes):
    """Collect route paths robustly across Starlette versions.

    Newer Starlette nests each included router as an ``_IncludedRouter`` in
    ``app.routes`` (which has no top-level ``.path``) instead of flattening to
    individual routes, so recurse into ``.routes`` to gather every path.
    """
    out = []
    for route in routes:
        path = getattr(route, "path", None)
        if isinstance(path, str):
            out.append(path)
        sub = getattr(route, "routes", None)
        if sub:
            out.extend(_all_route_paths(sub))
    return out


def test_legacy_auto_trader_routes_removed():
    paths = _all_route_paths(app.routes)
    assert all(not path.startswith("/api/auto-trader") for path in paths)
