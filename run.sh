#!/bin/bash
set -e

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

needs_setup() {
    if [ ! -d "backend/venv" ]; then
        return 0
    fi
    if [ ! -d "frontend/node_modules" ]; then
        return 0
    fi
    if [ ! -f ".setup-stamp.json" ]; then
        return 0
    fi

    python3 - <<'PY'
import hashlib
import json
import platform
import sys
from pathlib import Path

root = Path(".").resolve()
stamp_path = root / ".setup-stamp.json"

def sha256(path: Path) -> str:
    if not path.exists():
        return "missing"
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

try:
    stamp = json.loads(stamp_path.read_text(encoding="utf-8"))
except Exception:
    sys.exit(0)

current = {
    "python_version": platform.python_version(),
    "requirements_sha256": sha256(root / "backend" / "requirements.txt"),
    "requirements_trading_sha256": sha256(root / "backend" / "requirements-trading.txt"),
    "package_json_sha256": sha256(root / "frontend" / "package.json"),
    "package_lock_sha256": sha256(root / "frontend" / "package-lock.json"),
}

for key, value in current.items():
    if stamp.get(key) != value:
        sys.exit(0)

sys.exit(1)
PY
}

if needs_setup; then
    echo -e "${YELLOW}Setup missing or stale. Running setup...${NC}"
    ./setup.sh
fi

# Ensure TUI dependencies are installed
source backend/venv/bin/activate
python -c "import textual" 2>/dev/null || {
    echo -e "${CYAN}Installing TUI dependencies...${NC}"
    pip install -q textual rich
}

# Launch the TUI
exec python tui.py "$@"
