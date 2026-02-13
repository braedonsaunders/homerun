.PHONY: setup run dev clean help stop kill backend frontend build

# Default target
help:
	@echo "Autonomous Prediction Market Trading Platform"
	@echo ""
	@echo "Usage:"
	@echo "  make setup      - Install dependencies"
	@echo "  make run        - Start the application"
	@echo "  make dev        - Start in development mode"
	@echo "  make stop       - Stop all running services"
	@echo "  make restart    - Stop then start in dev mode"
	@echo "  make clean      - Remove generated files"
	@echo ""

# Kill processes on a port (usage: $(call kill-port,8000))
define kill-port
	@lsof -ti :$(1) 2>/dev/null | xargs kill -9 2>/dev/null || true
	@sleep 0.5
endef

# Setup dependencies
setup:
	@./setup.sh

# Run the application
run:
	@./run.sh

# Development mode (with hot reload)
dev: stop
	@echo "Starting in development mode..."
	@(cd backend && source venv/bin/activate && uvicorn main:app --reload --port 8000) &
	@(cd frontend && npm run dev)

# Stop all running services
stop:
	@echo "Stopping services on ports 8000 and 3000..."
	$(call kill-port,8000)
	$(call kill-port,3000)
	@echo "Ports cleared."

# Alias
kill: stop

# Restart everything
restart: stop dev

# Backend only
backend:
	$(call kill-port,8000)
	@cd backend && source venv/bin/activate && uvicorn main:app --reload --port 8000

# Frontend only
frontend:
	$(call kill-port,3000)
	@cd frontend && npm run dev

# Build frontend
build:
	@cd frontend && npm run build

# Clean generated files
clean: stop
	@echo "Cleaning generated files..."
	@rm -rf backend/venv
	@rm -rf backend/__pycache__
	@rm -rf backend/**/__pycache__
	@rm -rf frontend/node_modules
	@rm -rf frontend/dist
	@rm -rf data/*.db
	@echo "Clean complete"

# Install backend only
install-backend:
	@cd backend && python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt

# Install frontend only
install-frontend:
	@cd frontend && npm install
