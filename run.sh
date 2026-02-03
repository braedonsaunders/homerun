#!/bin/bash
set -e

echo "========================================="
echo "  Polymarket Arbitrage Scanner"
echo "========================================="

# Check if setup was run
if [ ! -d "backend/venv" ]; then
    echo "Setup not complete. Running setup first..."
    ./setup.sh
fi

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Shutting down..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start backend
echo ""
echo "Starting backend on http://localhost:8000..."
cd backend
source venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
cd ..

# Wait for backend to start
echo "Waiting for backend to start..."
sleep 3

# Check if backend started successfully
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "Warning: Backend may not have started correctly"
fi

# Start frontend
echo "Starting frontend on http://localhost:3000..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo ""
echo "========================================="
echo "  Application Running!"
echo "========================================="
echo ""
echo "  Dashboard: http://localhost:3000"
echo "  API:       http://localhost:8000"
echo "  API Docs:  http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Wait for processes
wait
