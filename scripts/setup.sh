#!/bin/bash
set -euo pipefail

# Navigate to project root (parent of scripts/)
cd "$(dirname "$0")/.."

REDIS_ONLY=0
POSTGRES_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --redis-only)
            REDIS_ONLY=1
            ;;
        --postgres-only)
            POSTGRES_ONLY=1
            ;;
    esac
done

run_with_optional_sudo() {
    if [ "$(id -u)" -eq 0 ]; then
        "$@"
        return
    fi
    if command -v sudo >/dev/null 2>&1; then
        sudo "$@"
        return
    fi
    "$@"
}

resolve_redis_server() {
    if command -v redis-server >/dev/null 2>&1; then
        command -v redis-server
        return 0
    fi

    if command -v brew >/dev/null 2>&1; then
        local brew_prefix
        brew_prefix="$(brew --prefix redis 2>/dev/null || true)"
        if [ -n "$brew_prefix" ] && [ -x "$brew_prefix/bin/redis-server" ]; then
            echo "$brew_prefix/bin/redis-server"
            return 0
        fi
    fi

    return 1
}

resolve_postgres_bin_dir() {
    if command -v initdb >/dev/null 2>&1 && command -v pg_ctl >/dev/null 2>&1; then
        dirname "$(command -v initdb)"
        return 0
    fi

    if command -v brew >/dev/null 2>&1; then
        local formula
        local brew_prefix
        for formula in postgresql@18 postgresql@17 postgresql@16 postgresql@15 postgresql@14 postgresql@13 postgresql@12 postgresql; do
            brew_prefix="$(brew --prefix "$formula" 2>/dev/null || true)"
            if [ -n "$brew_prefix" ] && [ -x "$brew_prefix/bin/initdb" ] && [ -x "$brew_prefix/bin/pg_ctl" ]; then
                echo "$brew_prefix/bin"
                return 0
            fi
        done
    fi

    return 1
}

ensure_redis_runtime() {
    if command -v docker >/dev/null 2>&1 || resolve_redis_server >/dev/null 2>&1; then
        echo "Redis runtime prerequisite found (docker or redis-server)."
        return 0
    fi

    echo "Redis runtime prerequisite missing. Attempting to install redis-server..."
    if command -v brew >/dev/null 2>&1; then
        if ! brew list redis >/dev/null 2>&1; then
            brew install redis
        fi
    elif command -v apt-get >/dev/null 2>&1; then
        run_with_optional_sudo apt-get update
        run_with_optional_sudo apt-get install -y redis-server
    elif command -v dnf >/dev/null 2>&1; then
        run_with_optional_sudo dnf install -y redis
    elif command -v yum >/dev/null 2>&1; then
        run_with_optional_sudo yum install -y redis
    elif command -v pacman >/dev/null 2>&1; then
        run_with_optional_sudo pacman -Sy --noconfirm redis
    else
        echo "Error: no supported package manager found to install redis-server."
        echo "Install Docker or redis-server manually, then rerun setup."
        return 1
    fi

    if command -v docker >/dev/null 2>&1 || resolve_redis_server >/dev/null 2>&1; then
        echo "Redis runtime prerequisite is now available."
        return 0
    fi

    echo "Error: automatic redis-server installation completed but redis-server is still unavailable."
    echo "Install Docker or redis-server manually, then rerun setup."
    return 1
}

has_postgres_runtime() {
    command -v docker >/dev/null 2>&1 || resolve_postgres_bin_dir >/dev/null 2>&1
}

ensure_postgres_runtime() {
    if has_postgres_runtime; then
        echo "Postgres runtime prerequisite found (docker or initdb+pg_ctl)."
        return 0
    fi

    echo "Postgres runtime prerequisite missing. Attempting to install PostgreSQL tools..."
    if command -v brew >/dev/null 2>&1; then
        if ! brew list postgresql@16 >/dev/null 2>&1 && ! brew list postgresql >/dev/null 2>&1; then
            brew install postgresql@16 || brew install postgresql
        fi
    elif command -v apt-get >/dev/null 2>&1; then
        run_with_optional_sudo apt-get update
        run_with_optional_sudo apt-get install -y postgresql
    elif command -v dnf >/dev/null 2>&1; then
        run_with_optional_sudo dnf install -y postgresql-server
    elif command -v yum >/dev/null 2>&1; then
        run_with_optional_sudo yum install -y postgresql-server
    elif command -v pacman >/dev/null 2>&1; then
        run_with_optional_sudo pacman -Sy --noconfirm postgresql
    else
        echo "Error: no supported package manager found to install PostgreSQL."
        echo "Install Docker or PostgreSQL tools (initdb + pg_ctl) manually, then rerun setup."
        return 1
    fi

    if has_postgres_runtime; then
        echo "Postgres runtime prerequisite is now available."
        return 0
    fi

    echo "Error: automatic PostgreSQL installation completed but initdb/pg_ctl is still unavailable."
    echo "Install Docker or PostgreSQL tools manually, then rerun setup."
    return 1
}

echo "========================================="
echo "  Autonomous Prediction Market Trading Platform Setup"
echo "========================================="
echo ""

if [ "$REDIS_ONLY" -eq 1 ] && [ "$POSTGRES_ONLY" -eq 1 ]; then
    ensure_redis_runtime
    ensure_postgres_runtime
    exit 0
fi

if [ "$REDIS_ONLY" -eq 1 ]; then
    ensure_redis_runtime
    exit 0
fi

if [ "$POSTGRES_ONLY" -eq 1 ]; then
    ensure_postgres_runtime
    exit 0
fi

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    echo "On Mac: brew install python@3.11"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PYTHON_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')
echo "Found Python $PYTHON_VERSION"

# Check Node.js
if ! command -v node &> /dev/null; then
    echo "Error: Node.js is required but not installed."
    echo "On Mac: brew install node"
    exit 1
fi

NODE_VERSION=$(node -v)
echo "Found Node.js $NODE_VERSION"

# Setup backend
echo ""
echo "Setting up backend..."
cd backend

if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing Python dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Check for OpenSSL/LibreSSL compatibility and attempt fix
SSL_LIB=$(python3 -c "import ssl; print(ssl.OPENSSL_VERSION)" 2>/dev/null || echo "unknown")
if echo "$SSL_LIB" | grep -qi "libressl"; then
    echo ""
    echo "Detected $SSL_LIB (macOS default)."
    echo "Installing pyopenssl for better SSL compatibility..."
    pip install -q pyopenssl cryptography 2>/dev/null || echo "  (pyopenssl install skipped - non-critical)"
fi

# Try to install trading dependencies (requires Python 3.10+)
if [ "$PYTHON_MINOR" -ge 10 ]; then
    echo "Installing trading dependencies..."
    pip install -q -r requirements-trading.txt 2>/dev/null || echo "  (trading deps skipped - optional)"
else
    echo ""
    echo "Note: Python 3.10+ required for live trading."
    echo "      Paper trading and scanning will work fine."
    echo "      Upgrade Python to enable live trading: brew install python@3.11"
fi

cd ..

# Setup frontend
echo ""
echo "Setting up frontend..."
cd frontend

echo "Installing Node.js dependencies..."
npm install --silent 2>/dev/null || npm install

cd ..

echo ""
echo "Setting up launcher tooling..."
export CXXFLAGS="${CXXFLAGS:-} -std=c++20"
npm --prefix scripts/tooling install --silent 2>/dev/null || npm --prefix scripts/tooling install

echo "Verifying PowerShell launcher syntax..."
node scripts/tooling/check_powershell_syntax.mjs scripts/run.ps1 scripts/setup.ps1

# Create data directory
mkdir -p data

echo ""
echo "Ensuring Redis runtime prerequisites..."
ensure_redis_runtime

echo "Ensuring Postgres runtime prerequisites..."
ensure_postgres_runtime

# Write setup fingerprint so run.sh can detect drift and auto-rerun setup.
if [ -x "backend/venv/bin/python" ]; then
    FINGERPRINT_PY_VERSION="$(backend/venv/bin/python -c 'import platform; print(platform.python_version())')"
else
    FINGERPRINT_PY_VERSION="$(python3 -c 'import platform; print(platform.python_version())')"
fi

SETUP_FINGERPRINT_PY_VERSION="$FINGERPRINT_PY_VERSION" python3 - <<'PY'
import hashlib
import json
import os
import platform
from pathlib import Path

root = Path(".").resolve()

def sha256(path: Path) -> str:
    if not path.exists():
        return "missing"
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

stamp = {
    "python_version": os.getenv("SETUP_FINGERPRINT_PY_VERSION", platform.python_version()),
    "requirements_sha256": sha256(root / "backend" / "requirements.txt"),
    "requirements_trading_sha256": sha256(root / "backend" / "requirements-trading.txt"),
    "package_json_sha256": sha256(root / "frontend" / "package.json"),
    "package_lock_sha256": sha256(root / "frontend" / "package-lock.json"),
    "launcher_tools_package_json_sha256": sha256(root / "scripts" / "tooling" / "package.json"),
    "launcher_tools_package_lock_sha256": sha256(root / "scripts" / "tooling" / "package-lock.json"),
}

(root / ".setup-stamp.json").write_text(json.dumps(stamp, indent=2), encoding="utf-8")
print("Wrote .setup-stamp.json")
PY

echo ""
echo "========================================="
echo "  Setup Complete!"
echo "========================================="
echo ""
echo "To start the application, run:"
echo "  ./scripts/run.sh"
echo ""
echo "Or run runtime validation only:"
echo "  ./scripts/run.sh --services-smoke-test"
echo ""
echo "Or start services individually:"
echo "  Backend:  cd backend && source venv/bin/activate && uvicorn main:app --reload"
echo "  Frontend: cd frontend && npm run dev"
echo ""
echo "The app will be available at:"
echo "  Frontend: http://localhost:3000"
echo "  Backend:  http://localhost:8000"
echo "  API Docs: http://localhost:8000/docs"
echo ""
