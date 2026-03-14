#!/bin/bash
set -euo pipefail

# Navigate to project root (grandparent of scripts/infra/)
cd "$(dirname "$0")/../.."

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
            echo "max_connections = 200"
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
    ./scripts/infra/setup.sh --postgres-only
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
    if ! tcp_ping "$POSTGRES_HOST" "$port"; then
        return 1
    fi
    [ "$port" = "$POSTGRES_PORT" ]
}

local_postgres_listener_port_for_data_dir() {
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
    if ! tcp_ping "$POSTGRES_HOST" "$port"; then
        return 1
    fi
    echo "$port"
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
    local existing_data_dir_port
    existing_data_dir_port="$(local_postgres_listener_port_for_data_dir 2>/dev/null || true)"
    if [ -n "$existing_data_dir_port" ]; then
        POSTGRES_PORT="$existing_data_dir_port"
        echo -e "${GREEN}Postgres already running from ${POSTGRES_DATA_DIR} on ${POSTGRES_HOST}:${POSTGRES_PORT}${NC}"
        return 0
    fi

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

cleanup_stale_homerun_processes() {
    # Kill orphaned Python worker processes from a previous crashed run.
    local project_root
    project_root="$(pwd)"
    local pids
    pids="$(ps -eo pid=,command= 2>/dev/null | grep -E 'workers\.host|workers\.runner|workers\.\w+_worker|uvicorn.*main:app|tui\.py' | grep -v grep | awk '{print $1}')" || true
    if [ -z "$pids" ]; then
        return 0
    fi
    local killed=0
    for pid in $pids; do
        if [ "$pid" = "$$" ]; then
            continue
        fi
        kill -9 "$pid" 2>/dev/null && killed=$((killed + 1)) || true
    done
    if [ "$killed" -gt 0 ]; then
        echo -e "${YELLOW}Cleaned up ${killed} stale Homerun process(es) from a previous run.${NC}"
        sleep 1
    fi
}

cleanup_local_postgres_if_owned() {
    local pg_bin_dir
    pg_bin_dir="$(resolve_postgres_bin_dir 2>/dev/null || true)"
    if [ -z "$pg_bin_dir" ] || [ ! -x "$pg_bin_dir/pg_ctl" ]; then
        return 0
    fi
    if local_postgres_listener_on_requested_port; then
        "$pg_bin_dir/pg_ctl" -D "$POSTGRES_DATA_DIR" -m fast -w stop >/dev/null 2>&1 || true
    fi
}

cleanup_started_services() {
    cleanup_stale_homerun_processes
    cleanup_started_postgres
    if [ -z "${DATABASE_URL_WAS_PROVIDED:-}" ]; then
        cleanup_local_postgres_if_owned
    fi
}

auto_update_repository() {
    if ! command -v git >/dev/null 2>&1; then
        echo -e "${YELLOW}Git is not installed; skipping repository auto-update.${NC}"
        return 0
    fi

    if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        return 0
    fi

    local branch
    branch="$(git branch --show-current 2>/dev/null || true)"
    if [ -z "$branch" ]; then
        echo -e "${YELLOW}Detached HEAD detected; skipping repository auto-update.${NC}"
        return 0
    fi

    local dirty
    dirty="$(git status --porcelain --untracked-files=no 2>/dev/null || true)"
    if [ -n "$dirty" ]; then
        echo -e "${YELLOW}Local tracked changes detected; skipping repository auto-update.${NC}"
        return 0
    fi

    local remote_name
    local remote_branch
    remote_name="origin"
    remote_branch="$branch"

    local upstream
    upstream="$(git rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null || true)"
    if [ -n "$upstream" ] && [[ "$upstream" == */* ]]; then
        remote_name="${upstream%%/*}"
        remote_branch="${upstream#*/}"
    else
        if ! git show-ref --verify --quiet "refs/remotes/${remote_name}/${remote_branch}"; then
            echo -e "${YELLOW}No upstream branch configured for '${branch}'; skipping repository auto-update.${NC}"
            return 0
        fi
    fi

    echo -e "${CYAN}Checking for code updates from ${remote_name}/${remote_branch}...${NC}"
    if ! git -c credential.interactive=never fetch --quiet "$remote_name" "$remote_branch" >/dev/null 2>&1; then
        echo -e "${YELLOW}Unable to fetch updates; continuing with local copy.${NC}"
        return 0
    fi

    local pull_output
    if pull_output="$(git -c credential.interactive=never pull --ff-only --no-rebase "$remote_name" "$remote_branch" 2>&1)"; then
        if echo "$pull_output" | grep -Eq "Already up[ -]to date\\."; then
            echo -e "${GREEN}Code is up to date.${NC}"
        else
            echo -e "${GREEN}Code updated from ${remote_name}/${remote_branch}.${NC}"
        fi
        return 0
    fi

    echo -e "${YELLOW}Auto-update skipped (non fast-forward or local commits). Continuing with local copy.${NC}"
    return 0
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
    "launcher_tools_package_json_sha256": sha256(root / "scripts" / "infra" / "tooling" / "package.json"),
    "launcher_tools_package_lock_sha256": sha256(root / "scripts" / "infra" / "tooling" / "package-lock.json"),
}

for key, value in current.items():
    if stamp.get(key) != value:
        sys.exit(0)

sys.exit(1)
PY
}

auto_update_repository

if needs_setup; then
    echo -e "${YELLOW}Setup missing or stale. Running setup...${NC}"
    ./scripts/infra/setup.sh
fi

# Kill orphaned workers from a previous crashed run before starting services.
cleanup_stale_homerun_processes

DATABASE_URL_WAS_PROVIDED="${DATABASE_URL:-}"

trap cleanup_started_services EXIT

if [ -n "${DATABASE_URL:-}" ]; then
    echo -e "${CYAN}Using provided DATABASE_URL; skipping launcher-managed Postgres startup.${NC}"
else
    ensure_postgres
    export DATABASE_URL="postgresql+asyncpg://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${POSTGRES_HOST}:${POSTGRES_PORT}/${POSTGRES_DB}"
fi

mkdir -p backend/.runtime
printf '%s\n' "$DATABASE_URL" > backend/.runtime/database_url

backend/venv/bin/python scripts/infra/ensure_postgres_ready.py --database-url "$DATABASE_URL"

# Ensure TUI dependencies are installed
source backend/venv/bin/activate
python -c "import textual" 2>/dev/null || {
    echo -e "${CYAN}Installing TUI dependencies...${NC}"
    PIP_USER=0 python -m pip install -q --no-user textual rich
}

if [ "$RUN_SERVICE_SMOKE_TEST" -eq 1 ]; then
    python scripts/infra/launcher_smoke.py
    exit $?
fi

# Launch the TUI
if [ "${#TUI_ARGS[@]}" -gt 0 ]; then
    python tui.py "${TUI_ARGS[@]}"
else
    python tui.py
fi
