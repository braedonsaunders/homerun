#!/bin/bash
set -e

echo "========================================="
echo "  Autonomous Prediction Market Trading Platform Setup"
echo "========================================="
echo ""

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    echo "On Mac: brew install python@3.11"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PYTHON_MAJOR=$(python3 -c 'import sys; print(sys.version_info.major)')
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

# Create .env if it doesn't exist
if [ ! -f .env ]; then
    echo ""
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "Created .env - edit this file to configure settings"
fi

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

# Create data directory
mkdir -p data

# Write setup fingerprint so run.sh can detect drift and auto-rerun setup.
python3 - <<'PY'
import hashlib
import json
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
    "python_version": platform.python_version(),
    "requirements_sha256": sha256(root / "backend" / "requirements.txt"),
    "requirements_trading_sha256": sha256(root / "backend" / "requirements-trading.txt"),
    "package_json_sha256": sha256(root / "frontend" / "package.json"),
    "package_lock_sha256": sha256(root / "frontend" / "package-lock.json"),
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
echo "  ./run.sh"
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
