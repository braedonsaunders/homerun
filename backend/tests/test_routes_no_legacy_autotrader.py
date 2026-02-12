import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from main import app


def test_legacy_auto_trader_routes_removed():
    paths = [route.path for route in app.routes]
    assert all(not path.startswith("/api/auto-trader") for path in paths)
