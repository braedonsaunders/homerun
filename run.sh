#!/bin/bash
set -e

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if setup was run
if [ ! -d "backend/venv" ]; then
    echo -e "${YELLOW}Setup not complete. Running setup first...${NC}"
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
