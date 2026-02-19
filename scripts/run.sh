#!/bin/bash
set -e

# Navigate to project root (parent of scripts/)
cd "$(dirname "$0")/.."

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

REDIS_HOST="${REDIS_HOST:-127.0.0.1}"
REDIS_PORT="${REDIS_PORT:-6379}"
REDIS_CONTAINER_NAME="${REDIS_CONTAINER_NAME:-homerun-redis}"
REDIS_IMAGE="${REDIS_IMAGE:-redis:7-alpine}"
REDIS_STARTED_BY_SCRIPT=0
REDIS_START_MODE=""

redis_ping() {
    python3 - "$REDIS_HOST" "$REDIS_PORT" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
payload = b"*1\r\n$4\r\nPING\r\n"

try:
    with socket.create_connection((host, port), timeout=0.5) as sock:
        sock.sendall(payload)
        sock.settimeout(0.5)
        data = sock.recv(64)
        if b"+PONG" in data:
            raise SystemExit(0)
except Exception:
    pass

raise SystemExit(1)
PY
}

redis_shutdown() {
    python3 - "$REDIS_HOST" "$REDIS_PORT" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
payload = b"*2\r\n$8\r\nSHUTDOWN\r\n$6\r\nNOSAVE\r\n"

try:
    with socket.create_connection((host, port), timeout=0.5) as sock:
        sock.sendall(payload)
except Exception:
    pass
PY
}

wait_for_redis() {
    for _ in $(seq 1 20); do
        if redis_ping; then
            return 0
        fi
        sleep 0.25
    done
    return 1
}

try_start_redis_docker() {
    if ! command -v docker >/dev/null 2>&1; then
        return 1
    fi
    if ! docker info >/dev/null 2>&1; then
        return 1
    fi

    if docker container inspect "$REDIS_CONTAINER_NAME" >/dev/null 2>&1; then
        docker start "$REDIS_CONTAINER_NAME" >/dev/null 2>&1 || return 1
    else
        docker run \
            --name "$REDIS_CONTAINER_NAME" \
            --detach \
            --publish "${REDIS_HOST}:${REDIS_PORT}:6379" \
            "$REDIS_IMAGE" \
            redis-server --save "" --appendonly no \
            >/dev/null 2>&1 || return 1
    fi
    return 0
}

try_start_redis_local() {
    if ! command -v redis-server >/dev/null 2>&1; then
        return 1
    fi
    redis-server \
        --bind "$REDIS_HOST" \
        --port "$REDIS_PORT" \
        --save "" \
        --appendonly no \
        --daemonize yes \
        >/dev/null 2>&1 || return 1
    return 0
}

has_redis_runtime() {
    command -v docker >/dev/null 2>&1 || command -v redis-server >/dev/null 2>&1
}

bootstrap_redis_runtime() {
    if has_redis_runtime; then
        return 0
    fi
    echo -e "${CYAN}Redis runtime missing; invoking setup redis bootstrap...${NC}"
    ./scripts/setup.sh --redis-only
}

cleanup_started_redis() {
    if [ "$REDIS_STARTED_BY_SCRIPT" -ne 1 ]; then
        return 0
    fi

    if [ "$REDIS_START_MODE" = "docker" ]; then
        if command -v docker >/dev/null 2>&1; then
            docker stop "$REDIS_CONTAINER_NAME" >/dev/null 2>&1 || true
        fi
        return 0
    fi

    if [ "$REDIS_START_MODE" = "local" ]; then
        if redis_ping; then
            redis_shutdown >/dev/null 2>&1 || true
        fi
        return 0
    fi
}

ensure_redis() {
    if redis_ping; then
        echo -e "${GREEN}Redis already running on ${REDIS_HOST}:${REDIS_PORT}${NC}"
        return 0
    fi

    bootstrap_redis_runtime

    echo -e "${CYAN}Starting Redis...${NC}"
    if try_start_redis_docker && wait_for_redis; then
        REDIS_STARTED_BY_SCRIPT=1
        REDIS_START_MODE="docker"
        echo -e "${GREEN}Redis started via Docker on ${REDIS_HOST}:${REDIS_PORT}${NC}"
        return 0
    fi

    if try_start_redis_local && wait_for_redis; then
        REDIS_STARTED_BY_SCRIPT=1
        REDIS_START_MODE="local"
        echo -e "${GREEN}Redis started via redis-server on ${REDIS_HOST}:${REDIS_PORT}${NC}"
        return 0
    fi

    echo -e "${YELLOW}Failed to start Redis automatically.${NC}"
    echo "Install Docker or redis-server, then rerun:"
    echo "  docker run --name ${REDIS_CONTAINER_NAME} -d -p ${REDIS_HOST}:${REDIS_PORT}:6379 ${REDIS_IMAGE}"
    exit 1
}

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
    ./scripts/setup.sh
fi

trap cleanup_started_redis EXIT

ensure_redis

# Ensure TUI dependencies are installed
source backend/venv/bin/activate
python -c "import textual" 2>/dev/null || {
    echo -e "${CYAN}Installing TUI dependencies...${NC}"
    pip install -q textual rich
}

# Launch the TUI
python tui.py "$@"
