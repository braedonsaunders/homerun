#!/bin/bash
set -euo pipefail

# Navigate to project root (grandparent of scripts/infra/)
cd "$(dirname "$0")/../.."

POSTGRES_ONLY=0
NO_BANNER=0
PYTHON_MIN_MINOR=10
PYTHON_MAX_MINOR=13
for arg in "$@"; do
    case "$arg" in
        --postgres-only)
            POSTGRES_ONLY=1
            ;;
        --no-banner)
            NO_BANNER=1
            ;;
    esac
done

# ── Colors ───────────────────────────────────────────────────────────

GREEN='\033[0;32m'
CYAN='\033[0;36m'
DARK_CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
WHITE='\033[1;37m'
DIM='\033[2m'
NC='\033[0m'

# ── Banner & UI Helpers ──────────────────────────────────────────────

show_banner() {
    echo ""
    echo -e "\033[38;2;30;107;69m    ██   ██  ██████  ███    ███ ███████ ██████  ██    ██ ███    ██\033[0m"
    echo -e "\033[38;2;35;138;85m    ██   ██ ██    ██ ████  ████ ██      ██   ██ ██    ██ ████   ██\033[0m"
    echo -e "\033[38;2;42;172;104m    ███████ ██    ██ ██ ████ ██ █████   ██████  ██    ██ ██ ██  ██\033[0m"
    echo -e "\033[38;2;53;212;122m    ██   ██ ██    ██ ██  ██  ██ ██      ██   ██ ██    ██ ██  ██ ██\033[0m"
    echo -e "\033[38;2;88;241;193m    ██   ██  ██████  ██      ██ ███████ ██   ██  ██████  ██   ████\033[0m"
    echo ""
    echo -e "${DIM}                    Autonomous Trading Platform${NC}"
    echo ""
}

STEP_NUM=0
TOTAL_STEPS=8

show_step() {
    STEP_NUM=$((STEP_NUM + 1))
    local text="$1"
    local text_len=${#text}
    local pad=$((46 - text_len))
    [ "$pad" -lt 2 ] && pad=2
    local dots
    dots=$(printf '.%.0s' $(seq 1 "$pad"))
    printf "    [%d/%d]  %s %s " "$STEP_NUM" "$TOTAL_STEPS" "$text" "$dots"
}

show_step_ok() {
    local detail="${1:-}"
    if [ -n "$detail" ]; then
        echo -e "${GREEN}OK  ${detail}${NC}"
    else
        echo -e "${GREEN}OK${NC}"
    fi
}

show_step_warn() {
    local detail="${1:-}"
    if [ -n "$detail" ]; then
        echo -e "${YELLOW}!!  ${detail}${NC}"
    else
        echo -e "${YELLOW}!!${NC}"
    fi
}

show_step_fail() {
    local detail="${1:-}"
    if [ -n "$detail" ]; then
        echo -e "${RED}FAIL  ${detail}${NC}"
    else
        echo -e "${RED}FAIL${NC}"
    fi
}

show_sub_info() {
    echo -e "${DIM}             $1${NC}"
}

# ── Utility Functions ────────────────────────────────────────────────

python_version_supported() {
    local python_cmd="$1"
    "$python_cmd" -c "import sys; raise SystemExit(0 if sys.version_info.major == 3 and $PYTHON_MIN_MINOR <= sys.version_info.minor <= $PYTHON_MAX_MINOR else 1)" >/dev/null 2>&1
}

find_supported_python() {
    local candidate
    for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
        if command -v "$candidate" >/dev/null 2>&1 && python_version_supported "$candidate"; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

install_supported_python() {
    show_sub_info "No Python 3.10-3.13 found. Attempting automatic install..."

    if command -v brew >/dev/null 2>&1; then
        brew install python@3.13 || brew install python@3.12 || brew install python@3.11
    elif command -v apt-get >/dev/null 2>&1; then
        run_with_optional_sudo apt-get update
        run_with_optional_sudo apt-get install -y python3.13 python3.13-venv || \
            run_with_optional_sudo apt-get install -y python3.12 python3.12-venv || \
            run_with_optional_sudo apt-get install -y python3.11 python3.11-venv || \
            run_with_optional_sudo apt-get install -y python3 python3-venv
    elif command -v dnf >/dev/null 2>&1; then
        run_with_optional_sudo dnf install -y python3.13 || \
            run_with_optional_sudo dnf install -y python3.12 || \
            run_with_optional_sudo dnf install -y python3.11 || \
            run_with_optional_sudo dnf install -y python3
    elif command -v yum >/dev/null 2>&1; then
        run_with_optional_sudo yum install -y python3.13 || \
            run_with_optional_sudo yum install -y python3.12 || \
            run_with_optional_sudo yum install -y python3.11 || \
            run_with_optional_sudo yum install -y python3
    elif command -v pacman >/dev/null 2>&1; then
        run_with_optional_sudo pacman -Sy --noconfirm python
    else
        return 1
    fi

    find_supported_python >/dev/null 2>&1
}

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

has_postgres_runtime() {
    command -v docker >/dev/null 2>&1 || resolve_postgres_bin_dir >/dev/null 2>&1
}

ensure_postgres_runtime() {
    if has_postgres_runtime; then
        if command -v docker >/dev/null 2>&1; then
            echo "docker"
        else
            echo "initdb+pg_ctl"
        fi
        return 0
    fi

    show_sub_info "Postgres runtime missing. Attempting to install PostgreSQL tools..."
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
        return 1
    fi

    if has_postgres_runtime; then
        echo "installed"
        return 0
    fi

    return 1
}

# ── WeasyPrint native deps (Pango/Cairo/GdkPixbuf) ──────────────────
# WeasyPrint powers the executive PDF report endpoint
# (services/reports/wallet_strategy_report.py).  Without these libs the
# report endpoint returns 503 with an install hint, but the rest of the
# API runs fine — we therefore best-effort install and don't fail setup
# if the package manager isn't available.
ensure_weasyprint_native_deps() {
    if command -v brew >/dev/null 2>&1; then
        for pkg in pango cairo gdk-pixbuf libffi; do
            brew list "$pkg" >/dev/null 2>&1 || brew install "$pkg" >/dev/null 2>&1 || true
        done
        return 0
    fi
    if command -v apt-get >/dev/null 2>&1; then
        run_with_optional_sudo apt-get update >/dev/null 2>&1 || true
        run_with_optional_sudo apt-get install -y \
            libpango-1.0-0 libpangoft2-1.0-0 libharfbuzz0b \
            libfreetype6 libgdk-pixbuf-2.0-0 libffi-dev \
            fonts-dejavu-core >/dev/null 2>&1 || true
        return 0
    fi
    if command -v dnf >/dev/null 2>&1; then
        run_with_optional_sudo dnf install -y \
            pango cairo gdk-pixbuf2 libffi dejavu-sans-fonts >/dev/null 2>&1 || true
        return 0
    fi
    if command -v pacman >/dev/null 2>&1; then
        run_with_optional_sudo pacman -Sy --noconfirm \
            pango cairo gdk-pixbuf2 libffi ttf-dejavu >/dev/null 2>&1 || true
        return 0
    fi
    # No supported package manager — print a hint, keep going.
    echo "  (skipping WeasyPrint native deps install — no supported package manager found.)"
    return 0
}

# ── PostgresOnly Mode ────────────────────────────────────────────────

if [ "$POSTGRES_ONLY" -eq 1 ]; then
    pg_result="$(ensure_postgres_runtime 2>/dev/null || true)"
    if [ -z "$pg_result" ]; then
        echo "Error: no supported package manager found to install PostgreSQL."
        echo "Install Docker or PostgreSQL tools (initdb + pg_ctl) manually, then rerun setup."
        exit 1
    fi
    exit 0
fi

# ── Main Setup Flow ─────────────────────────────────────────────────

if [ "$NO_BANNER" -eq 0 ]; then
    show_banner
fi

# ── Step 1: Detect Python ───────────────────────────────────────────

show_step "Detecting Python"

PYTHON_CMD="$(find_supported_python || true)"
if [ -z "$PYTHON_CMD" ]; then
    install_supported_python || true
    PYTHON_CMD="$(find_supported_python || true)"
fi
if [ -z "$PYTHON_CMD" ]; then
    show_step_fail "Python 3.10-3.13 required"
    if command -v python3 >/dev/null 2>&1; then
        show_sub_info "Detected: $(python3 -c 'import platform; print(platform.python_version())')"
    fi
    show_sub_info "Install Python 3.12 or 3.11 (with venv support) and rerun setup."
    exit 1
fi
PYTHON_VERSION="$($PYTHON_CMD -c 'import platform; print(platform.python_version())')"
show_step_ok "$PYTHON_VERSION ($PYTHON_CMD)"

# ── Step 2: Detect Node.js ──────────────────────────────────────────

show_step "Detecting Node.js"

if ! command -v node &> /dev/null; then
    show_step_fail "not found"
    show_sub_info "On Mac: brew install node"
    exit 1
fi

NODE_VERSION=$(node -v)
show_step_ok "$NODE_VERSION"

# ── Step 3: Backend environment ──────────────────────────────────────

show_step "Preparing backend environment"

cd backend

if [ -d "venv" ]; then
    if [ ! -x "venv/bin/python" ]; then
        rm -rf venv
    elif ! venv/bin/python -c "import sys; raise SystemExit(0 if sys.version_info.major == 3 and $PYTHON_MIN_MINOR <= sys.version_info.minor <= $PYTHON_MAX_MINOR else 1)" >/dev/null 2>&1; then
        rm -rf venv
    fi
fi

if [ ! -d "venv" ]; then
    "$PYTHON_CMD" -m venv venv
fi

source venv/bin/activate

VENV_PYTHON_VERSION="$(python -c 'import platform; print(platform.python_version())')"
if ! python -c "import sys; raise SystemExit(0 if sys.version_info.major == 3 and $PYTHON_MIN_MINOR <= sys.version_info.minor <= $PYTHON_MAX_MINOR else 1)" >/dev/null 2>&1; then
    cd ..
    show_step_fail "venv Python $VENV_PYTHON_VERSION unsupported"
    show_sub_info "Delete backend/venv and rerun setup."
    exit 1
fi

show_step_ok "venv ready"

# ── Step 4a: WeasyPrint native libs (best-effort, no failure) ───────
# Pango / Cairo / GdkPixbuf are required at runtime by WeasyPrint
# (services/reports/wallet_strategy_report.py).  The PDF report
# endpoint is the only consumer; if these libs are missing the rest
# of the API still runs and the endpoint returns a 503 with an
# install hint.  Best-effort install here so the typical "git clone
# && setup.sh" path produces a working report endpoint out of the
# box on Mac (brew) and Linux (apt/dnf/pacman).
ensure_weasyprint_native_deps

# ── Step 4: Python dependencies ──────────────────────────────────────

show_step "Installing Python dependencies"

PIP_USER=0 python -m pip install -q --no-user --upgrade pip
PIP_USER=0 python -m pip install -q --no-user -r requirements.txt
if ! python -c "import socksio" >/dev/null 2>&1; then
    show_step_fail "socksio missing"
    show_sub_info "Run: python -m pip install --no-user \"httpx[socks]>=0.27.0,<1.0\""
    exit 1
fi

# Check for OpenSSL/LibreSSL compatibility and attempt fix
SSL_LIB=$(python -c "import ssl; print(ssl.OPENSSL_VERSION)" 2>/dev/null || echo "unknown")
if echo "$SSL_LIB" | grep -qi "libressl"; then
    PIP_USER=0 python -m pip install -q --no-user pyopenssl cryptography 2>/dev/null || true
fi

show_step_ok

# ── Step 5: Trading dependencies ─────────────────────────────────────

show_step "Installing trading packages"

PIP_USER=0 python -m pip install -q --no-user -r requirements-trading.txt
if ! python -c "import py_clob_client_v2, eth_account" >/dev/null 2>&1; then
    show_step_fail "trading imports missing"
    show_sub_info "Expected: py_clob_client_v2, eth_account"
    exit 1
fi

cd ..
show_step_ok

# ── Step 6: Frontend dependencies ────────────────────────────────────

show_step "Installing frontend packages"

cd frontend
npm install --silent 2>/dev/null || npm install
cd ..

show_step_ok

# ── Step 7: Launcher tooling ────────────────────────────────────────

show_step "Setting up launcher tooling"

export CXXFLAGS="${CXXFLAGS:-} -std=c++20"
TOOLING_OK=0
npm --prefix scripts/infra/tooling install --silent 2>/dev/null || npm --prefix scripts/infra/tooling install 2>/dev/null || true
if [ -d "scripts/infra/tooling/node_modules/tree-sitter" ]; then
    TOOLING_OK=1
fi

if [ "$TOOLING_OK" -eq 1 ]; then
    if node scripts/infra/tooling/check_powershell_syntax.mjs scripts/infra/run.ps1 scripts/infra/setup.ps1 2>/dev/null; then
        show_step_ok
    else
        show_step_warn "syntax check failed (non-fatal)"
    fi
else
    show_step_warn "skipped (non-fatal)"
fi

# ── Step 8: Postgres runtime ────────────────────────────────────────

mkdir -p data

show_step "Verifying Postgres runtime"

pg_result="$(ensure_postgres_runtime 2>/dev/null || true)"
if [ -z "$pg_result" ]; then
    show_step_fail "no runtime found"
    show_sub_info "Install Docker or PostgreSQL tools (initdb + pg_ctl), then rerun setup."
    exit 1
fi

show_step_ok "$pg_result"

# ── Write setup fingerprint ─────────────────────────────────────────

if [ -x "backend/venv/bin/python" ]; then
    FINGERPRINT_PY_VERSION="$(backend/venv/bin/python -c 'import platform; print(platform.python_version())')"
else
    FINGERPRINT_PY_VERSION="$($PYTHON_CMD -c 'import platform; print(platform.python_version())')"
fi

SETUP_FINGERPRINT_PY_VERSION="$FINGERPRINT_PY_VERSION" "$PYTHON_CMD" - <<'PY'
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
    "launcher_tools_package_json_sha256": sha256(root / "scripts" / "infra" / "tooling" / "package.json"),
    "launcher_tools_package_lock_sha256": sha256(root / "scripts" / "infra" / "tooling" / "package-lock.json"),
}

(root / ".setup-stamp.json").write_text(json.dumps(stamp, indent=2), encoding="utf-8")
PY

# ── Completion ───────────────────────────────────────────────────────

echo ""
echo -e "${DIM}    ─────────────────────────────────────────────────────────────────${NC}"
echo ""
echo -e "${GREEN}    Setup complete!${NC}"
echo ""
echo -e "${WHITE}    Start the application:${NC}"
echo -e "${CYAN}      ./scripts/infra/run.sh${NC}"
echo ""
echo -e "${DIM}    Or start services individually:${NC}"
echo -e "${DIM}      Backend:  cd backend && source venv/bin/activate && uvicorn main:app --reload${NC}"
echo -e "${DIM}      Frontend: cd frontend && npm run dev${NC}"
echo ""
echo -e "${WHITE}    Endpoints:${NC}"
echo -e "${CYAN}      Frontend  http://localhost:3000${NC}"
echo -e "${CYAN}      Backend   http://localhost:8000${NC}"
echo -e "${CYAN}      API Docs  http://localhost:8000/docs${NC}"
echo ""
echo -e "${DIM}    ─────────────────────────────────────────────────────────────────${NC}"
echo ""
