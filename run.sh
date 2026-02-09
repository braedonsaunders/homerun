#!/bin/bash
set -e

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}"
cat << 'EOF'
 _   _  ___  __  __ _____ ____  _   _ _   _
| | | |/ _ \|  \/  | ____|  _ \| | | | \ | |
| |_| | | | | |\/| |  _| | |_) | | | |  \| |
|  _  | |_| | |  | | |___|  _ <| |_| | |\  |
|_| |_|\___/|_|  |_|_____|_| \_\\___/|_| \_|
EOF
echo -e "${NC}"
echo -e "${CYAN}Autonomous Prediction Market Trading Platform${NC}"
echo ""

# Check if setup was run
if [ ! -d "backend/venv" ]; then
    echo -e "${YELLOW}Setup not complete. Running setup first...${NC}"
    ./setup.sh
fi

# Kill any process using a given port
kill_port() {
    local port=$1
    local pids
    pids=$(lsof -ti :"$port" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo -e "${YELLOW}Killing existing process(es) on port $port (PIDs: $pids)${NC}"
        echo "$pids" | xargs kill -9 2>/dev/null || true
        sleep 1
    fi
}

# Function to cleanup on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down...${NC}"
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    exit 0
}

trap cleanup SIGINT SIGTERM

# Clean up ports before starting
kill_port 8000
kill_port 3000

# Start backend
echo -e "${CYAN}Starting backend...${NC}"
cd backend
source venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
cd ..

# Wait for backend to start
sleep 3

# Check if backend started successfully
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo -e "${YELLOW}Warning: Backend may not have started correctly${NC}"
fi

# Start frontend
echo -e "${CYAN}Starting frontend...${NC}"
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo ""
echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}  HOMERUN is running!${NC}"
echo -e "${GREEN}=========================================${NC}"
echo ""
echo -e "  Dashboard: ${CYAN}http://localhost:3000${NC}"
echo -e "  API:       ${CYAN}http://localhost:8000${NC}"
echo -e "  API Docs:  ${CYAN}http://localhost:8000/docs${NC}"
echo ""
echo -e "Press ${YELLOW}Ctrl+C${NC} to stop"
echo ""

# Wait for processes
wait
