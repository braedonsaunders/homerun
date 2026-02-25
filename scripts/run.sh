#!/bin/bash
set -euo pipefail

# Navigate to project root (parent of scripts/)
cd "$(dirname "$0")/.."

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

RUN_SERVICE_SMOKE_TEST=0
TUI_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --services-smoke-test)
            RUN_SERVICE_SMOKE_TEST=1
            ;;
        *)
            TUI_ARGS+=("$arg")
            ;;
    esac
done

REDIS_HOST="${REDIS_HOST:-127.0.0.1}"
REDIS_PORT="${REDIS_PORT:-6379}"
REDIS_CONTAINER_NAME="${REDIS_CONTAINER_NAME:-homerun-redis}"
REDIS_IMAGE="${REDIS_IMAGE:-redis:7-alpine}"
REDIS_STARTED_BY_SCRIPT=0
REDIS_START_MODE=""
REDIS_DOCKER_CREATED_BY_SCRIPT=0

POSTGRES_HOST="${POSTGRES_HOST:-127.0.0.1}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_DB="${POSTGRES_DB:-homerun}"
POSTGRES_USER="${POSTGRES_USER:-homerun}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-homerun}"
POSTGRES_CONTAINER_NAME="${POSTGRES_CONTAINER_NAME:-homerun-postgres}"
POSTGRES_IMAGE="${POSTGRES_IMAGE:-postgres:16-alpine}"
POSTGRES_DATA_DIR="${POSTGRES_DATA_DIR:-$(pwd)/data/postgres}"
POSTGRES_STARTED_BY_SCRIPT=0
POSTGRES_START_MODE=""
POSTGRES_DOCKER_CREATED_BY_SCRIPT=0

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

tcp_ping() {
    python3 - "$1" "$2" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])

try:
    with socket.create_connection((host, port), timeout=0.5):
        raise SystemExit(0)
except Exception:
    raise SystemExit(1)
PY
}

wait_for_service() {
    local host="$1"
    local port="$2"
    for _ in $(seq 1 40); do
        if tcp_ping "$host" "$port"; then
            return 0
        fi
        sleep 0.25
    done
    return 1
}

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
        REDIS_DOCKER_CREATED_BY_SCRIPT=1
    fi
    return 0
}

try_start_redis_local() {
    local redis_server
    redis_server="$(resolve_redis_server 2>/dev/null || true)"
    if [ -z "$redis_server" ]; then
        return 1
    fi

    "$redis_server" \
        --bind "$REDIS_HOST" \
        --port "$REDIS_PORT" \
        --save "" \
        --appendonly no \
        --daemonize yes \
        >/dev/null 2>&1 || return 1
    return 0
}

has_redis_runtime() {
    command -v docker >/dev/null 2>&1 || resolve_redis_server >/dev/null 2>&1
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
            if [ "$REDIS_DOCKER_CREATED_BY_SCRIPT" -eq 1 ]; then
                docker rm "$REDIS_CONTAINER_NAME" >/dev/null 2>&1 || true
            fi
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
    if try_start_redis_docker && wait_for_service "$REDIS_HOST" "$REDIS_PORT"; then
        REDIS_STARTED_BY_SCRIPT=1
        REDIS_START_MODE="docker"
        echo -e "${GREEN}Redis started via Docker on ${REDIS_HOST}:${REDIS_PORT}${NC}"
        return 0
    fi

    if try_start_redis_local && wait_for_service "$REDIS_HOST" "$REDIS_PORT"; then
        REDIS_STARTED_BY_SCRIPT=1
        REDIS_START_MODE="local"
        echo -e "${GREEN}Redis started via redis-server on ${REDIS_HOST}:${REDIS_PORT}${NC}"
        return 0
    fi

    echo -e "${YELLOW}Failed to start Redis automatically.${NC}"
    echo "Install Docker or redis-server, then rerun."
    exit 1
}

postgres_ping() {
    tcp_ping "$POSTGRES_HOST" "$POSTGRES_PORT"
}

try_start_postgres_docker() {
    if ! command -v docker >/dev/null 2>&1; then
        return 1
    fi
    if ! docker info >/dev/null 2>&1; then
        return 1
    fi

    if docker container inspect "$POSTGRES_CONTAINER_NAME" >/dev/null 2>&1; then
        docker start "$POSTGRES_CONTAINER_NAME" >/dev/null 2>&1 || return 1
    else
        mkdir -p "$POSTGRES_DATA_DIR"
        docker run \
            --name "$POSTGRES_CONTAINER_NAME" \
            --detach \
            --publish "${POSTGRES_HOST}:${POSTGRES_PORT}:5432" \
            --env "POSTGRES_DB=${POSTGRES_DB}" \
            --env "POSTGRES_USER=${POSTGRES_USER}" \
            --env "POSTGRES_PASSWORD=${POSTGRES_PASSWORD}" \
            --volume "${POSTGRES_DATA_DIR}:/var/lib/postgresql/data" \
            "$POSTGRES_IMAGE" \
            >/dev/null 2>&1 || return 1
        POSTGRES_DOCKER_CREATED_BY_SCRIPT=1
    fi

    return 0
}

try_start_postgres_local() {
    local pg_bin_dir
    local initdb_bin
    local pg_ctl_bin

    pg_bin_dir="$(resolve_postgres_bin_dir 2>/dev/null || true)"
    if [ -z "$pg_bin_dir" ]; then
        return 1
    fi

    initdb_bin="$pg_bin_dir/initdb"
    pg_ctl_bin="$pg_bin_dir/pg_ctl"

    mkdir -p "$POSTGRES_DATA_DIR"

    if [ ! -f "$POSTGRES_DATA_DIR/PG_VERSION" ]; then
        "$initdb_bin" -D "$POSTGRES_DATA_DIR" -U "$POSTGRES_USER" --encoding=UTF8 --locale=C >/dev/null 2>&1 || return 1
        cat > "$POSTGRES_DATA_DIR/pg_hba.conf" <<HBA
local all all trust
host all all 127.0.0.1/32 trust
host all all ::1/128 trust
HBA
        {
            echo "listen_addresses = '${POSTGRES_HOST}'"
            echo "port = ${POSTGRES_PORT}"
        } >> "$POSTGRES_DATA_DIR/postgresql.conf"
    fi

    # Start without -w (don't let pg_ctl block waiting).
    # The caller's wait_for_service handles readiness polling.
    # On some platforms a hard-killed Postgres can leave stale shared-memory;
    # if the first attempt fails we wait briefly and retry once.
    if "$pg_ctl_bin" -D "$POSTGRES_DATA_DIR" -o "-h ${POSTGRES_HOST} -p ${POSTGRES_PORT}" start >/dev/null 2>&1; then
        return 0
    fi

    # Stale shared-memory – remove PID file, pause, retry.
    rm -f "$POSTGRES_DATA_DIR/postmaster.pid"
    sleep 3
    "$pg_ctl_bin" -D "$POSTGRES_DATA_DIR" -o "-h ${POSTGRES_HOST} -p ${POSTGRES_PORT}" start >/dev/null 2>&1 || return 1
    return 0
}

has_postgres_runtime() {
    command -v docker >/dev/null 2>&1 || resolve_postgres_bin_dir >/dev/null 2>&1
}

bootstrap_postgres_runtime() {
    if has_postgres_runtime; then
        return 0
    fi
    echo -e "${CYAN}Postgres runtime missing; invoking setup postgres bootstrap...${NC}"
    ./scripts/setup.sh --postgres-only
}

cleanup_started_postgres() {
    if [ "$POSTGRES_STARTED_BY_SCRIPT" -ne 1 ]; then
        return 0
    fi

    if [ "$POSTGRES_START_MODE" = "docker" ]; then
        if command -v docker >/dev/null 2>&1; then
            docker stop "$POSTGRES_CONTAINER_NAME" >/dev/null 2>&1 || true
            if [ "$POSTGRES_DOCKER_CREATED_BY_SCRIPT" -eq 1 ]; then
                docker rm "$POSTGRES_CONTAINER_NAME" >/dev/null 2>&1 || true
            fi
        fi
        return 0
    fi

    if [ "$POSTGRES_START_MODE" = "local" ]; then
        local pg_bin_dir
        pg_bin_dir="$(resolve_postgres_bin_dir 2>/dev/null || true)"
        if [ -n "$pg_bin_dir" ] && [ -x "$pg_bin_dir/pg_ctl" ]; then
            "$pg_bin_dir/pg_ctl" -D "$POSTGRES_DATA_DIR" -m fast -w stop >/dev/null 2>&1 || true
        fi
    fi
}

docker_postgres_listener_on_requested_port() {
    if ! command -v docker >/dev/null 2>&1; then
        return 1
    fi
    if ! docker container inspect "$POSTGRES_CONTAINER_NAME" >/dev/null 2>&1; then
        return 1
    fi
    local running
    running="$(docker inspect -f '{{.State.Running}}' "$POSTGRES_CONTAINER_NAME" 2>/dev/null || true)"
    if [ "$running" != "true" ]; then
        return 1
    fi
    local host_port
    host_port="$(docker inspect -f '{{with index .NetworkSettings.Ports "5432/tcp"}}{{(index . 0).HostPort}}{{end}}' "$POSTGRES_CONTAINER_NAME" 2>/dev/null || true)"
    [ "$host_port" = "$POSTGRES_PORT" ]
}

local_postgres_listener_on_requested_port() {
    local pid_file="$POSTGRES_DATA_DIR/postmaster.pid"
    if [ ! -f "$pid_file" ]; then
        return 1
    fi

    local pid
    local port
    pid="$(sed -n '1p' "$pid_file" 2>/dev/null || true)"
    port="$(sed -n '4p' "$pid_file" 2>/dev/null || true)"
    if [ -z "$pid" ] || [ -z "$port" ]; then
        return 1
    fi
    if ! kill -0 "$pid" >/dev/null 2>&1; then
        return 1
    fi
    [ "$port" = "$POSTGRES_PORT" ]
}

postgres_listener_owned_by_launcher() {
    docker_postgres_listener_on_requested_port || local_postgres_listener_on_requested_port
}

find_available_postgres_port() {
    local start_port="$1"
    local max_port=$((start_port + 32))
    local port
    for port in $(seq "$start_port" "$max_port"); do
        if ! tcp_ping "$POSTGRES_HOST" "$port"; then
            echo "$port"
            return 0
        fi
    done
    return 1
}

ensure_postgres() {
    if postgres_ping; then
        if postgres_listener_owned_by_launcher; then
            echo -e "${GREEN}Postgres already running on ${POSTGRES_HOST}:${POSTGRES_PORT}${NC}"
            return 0
        fi
        local requested_port="$POSTGRES_PORT"
        local discovered_port
        discovered_port="$(find_available_postgres_port "$((POSTGRES_PORT + 1))" 2>/dev/null || true)"
        if [ -z "$discovered_port" ]; then
            echo -e "${YELLOW}Port ${POSTGRES_PORT} is occupied by a non-launcher service and no alternate Postgres port was found.${NC}"
            echo "Set DATABASE_URL manually or free up a local port, then rerun."
            exit 1
        fi
        POSTGRES_PORT="$discovered_port"
        echo -e "${YELLOW}Port ${requested_port} is in use by a non-launcher service. Launching project Postgres on ${POSTGRES_PORT} instead.${NC}"
    fi

    bootstrap_postgres_runtime

    echo -e "${CYAN}Starting Postgres...${NC}"
    if try_start_postgres_docker && wait_for_service "$POSTGRES_HOST" "$POSTGRES_PORT"; then
        POSTGRES_STARTED_BY_SCRIPT=1
        POSTGRES_START_MODE="docker"
        echo -e "${GREEN}Postgres started via Docker on ${POSTGRES_HOST}:${POSTGRES_PORT}${NC}"
        return 0
    fi

    if try_start_postgres_local && wait_for_service "$POSTGRES_HOST" "$POSTGRES_PORT"; then
        POSTGRES_STARTED_BY_SCRIPT=1
        POSTGRES_START_MODE="local"
        echo -e "${GREEN}Postgres started via local postgres on ${POSTGRES_HOST}:${POSTGRES_PORT}${NC}"
        return 0
    fi

    echo -e "${YELLOW}Failed to start Postgres automatically.${NC}"
    echo "Install Docker or PostgreSQL tools (initdb + pg_ctl), then rerun."
    exit 1
}

cleanup_started_services() {
    cleanup_started_postgres
    cleanup_started_redis
}

needs_setup() {
    if [ ! -d "backend/venv" ]; then
        return 0
    fi
    if [ ! -x "backend/venv/bin/python" ]; then
        return 0
    fi
    if [ ! -d "frontend/node_modules" ]; then
        return 0
    fi
    if [ ! -f ".setup-stamp.json" ]; then
        return 0
    fi

    if ! backend/venv/bin/python -c 'import sys; raise SystemExit(0 if sys.version_info.major == 3 and 10 <= sys.version_info.minor <= 13 else 1)' >/dev/null 2>&1; then
        return 0
    fi
    if ! backend/venv/bin/python -c "import py_clob_client, eth_account" >/dev/null 2>&1; then
        return 0
    fi

    local fingerprint_python_version
    if [ -x "backend/venv/bin/python" ]; then
        fingerprint_python_version="$(backend/venv/bin/python -c 'import platform; print(platform.python_version())')"
    else
        fingerprint_python_version="$(python3 -c 'import platform; print(platform.python_version())')"
    fi

    SETUP_FINGERPRINT_PY_VERSION="$fingerprint_python_version" python3 - <<'PY'
import hashlib
import json
import os
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
    "python_version": os.getenv("SETUP_FINGERPRINT_PY_VERSION", platform.python_version()),
    "requirements_sha256": sha256(root / "backend" / "requirements.txt"),
    "requirements_trading_sha256": sha256(root / "backend" / "requirements-trading.txt"),
    "package_json_sha256": sha256(root / "frontend" / "package.json"),
    "package_lock_sha256": sha256(root / "frontend" / "package-lock.json"),
    "launcher_tools_package_json_sha256": sha256(root / "scripts" / "tooling" / "package.json"),
    "launcher_tools_package_lock_sha256": sha256(root / "scripts" / "tooling" / "package-lock.json"),
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

trap cleanup_started_services EXIT

ensure_redis

if [ -n "${DATABASE_URL:-}" ]; then
    echo -e "${CYAN}Using provided DATABASE_URL; skipping launcher-managed Postgres startup.${NC}"
else
    ensure_postgres
    export DATABASE_URL="postgresql+asyncpg://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${POSTGRES_HOST}:${POSTGRES_PORT}/${POSTGRES_DB}"
fi

backend/venv/bin/python scripts/ensure_postgres_ready.py --database-url "$DATABASE_URL"

# Ensure TUI dependencies are installed
source backend/venv/bin/activate
python -c "import textual" 2>/dev/null || {
    echo -e "${CYAN}Installing TUI dependencies...${NC}"
    PIP_USER=0 python -m pip install -q --no-user textual rich
}

if [ "$RUN_SERVICE_SMOKE_TEST" -eq 1 ]; then
    python scripts/launcher_smoke.py
    exit $?
fi

# Launch the TUI
if [ "${#TUI_ARGS[@]}" -gt 0 ]; then
    python tui.py "${TUI_ARGS[@]}"
else
    python tui.py
fi
