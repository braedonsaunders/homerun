.PHONY: setup run dev docker docker-down clean help

# Default target
help:
	@echo "Polymarket Arbitrage Scanner"
	@echo ""
	@echo "Usage:"
	@echo "  make setup      - Install dependencies"
	@echo "  make run        - Start the application"
	@echo "  make dev        - Start in development mode"
	@echo "  make docker     - Run with Docker Compose"
	@echo "  make docker-down- Stop Docker containers"
	@echo "  make clean      - Remove generated files"
	@echo ""

# Setup dependencies
setup:
	@./setup.sh

# Run the application
run:
	@./run.sh

# Development mode (with hot reload)
dev:
	@echo "Starting in development mode..."
	@(cd backend && source venv/bin/activate && uvicorn main:app --reload --port 8000) &
	@(cd frontend && npm run dev)

# Docker Compose
docker:
	@echo "Starting with Docker Compose..."
	@docker-compose up --build

docker-down:
	@docker-compose down

# Backend only
backend:
	@cd backend && source venv/bin/activate && uvicorn main:app --reload --port 8000

# Frontend only
frontend:
	@cd frontend && npm run dev

# Build frontend
build:
	@cd frontend && npm run build

# Clean generated files
clean:
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
