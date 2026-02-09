# Windows Deployment Guide

This guide covers setting up, running, and updating Homerun on Windows.

---

## Prerequisites

- **Python 3.10+** — [Download](https://www.python.org/downloads/) (check "Add Python to PATH" during install)
- **Node.js 18+** — [Download](https://nodejs.org/)
- **Git** — [Download](https://git-scm.com/download/win)

Verify installation:

```powershell
python --version
node --version
git --version
```

> **Note:** On Windows, use `python` instead of `python3`. If `python` opens the Microsoft Store, see [Troubleshooting](#troubleshooting).

---

## Option 1: PowerShell Scripts (Recommended)

### First-Time Setup

```powershell
git clone https://github.com/braedonsaunders/homerun.git
cd homerun
.\setup.ps1
```

### Run

```powershell
.\run.ps1
```

### Update

```powershell
# 1. Stop the app (Ctrl+C if running)

# 2. Pull latest code
git pull origin main

# 3. Re-run setup
.\setup.ps1

# 4. Launch
.\run.ps1
```

---

## Option 2: Manual Setup

### 1. Clone and Configure

```powershell
git clone https://github.com/braedonsaunders/homerun.git
cd homerun
copy .env.example .env
```

Edit `.env` with your settings (use Notepad, VS Code, etc.).

### 2. Backend Setup

```powershell
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

# Optional: trading dependencies (Python 3.10+ required)
pip install -r requirements-trading.txt
```

### 3. Frontend Setup

```powershell
cd ..\frontend
npm install
```

### 4. Run

Open **two separate terminals**:

**Terminal 1 — Backend:**
```powershell
cd backend
.\venv\Scripts\Activate.ps1
uvicorn main:app --host 0.0.0.0 --port 8000
```

**Terminal 2 — Frontend:**
```powershell
cd frontend
npm run dev
```

### 5. Update (Manual)

```powershell
# Stop both terminals (Ctrl+C)

git pull origin main

# Re-install backend dependencies
cd backend
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Re-install frontend dependencies
cd ..\frontend
npm install

# Start again (two terminals as above)
```

---

## Option 3: Docker (Cross-Platform)

Docker Desktop for Windows works identically to Linux/macOS.

### Setup

1. Install [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/)
2. Ensure WSL 2 backend is enabled (Docker Desktop will prompt you)

### Run

```powershell
git clone https://github.com/braedonsaunders/homerun.git
cd homerun
docker-compose up --build
```

### Update

```powershell
git pull origin main
docker-compose down
docker-compose up --build
```

---

## Services

| Service | URL |
|---------|-----|
| Dashboard | `http://localhost:3000` |
| API | `http://localhost:8000` |
| Swagger Docs | `http://localhost:8000/docs` |
| WebSocket | `ws://localhost:8000/ws` |

---

## Database

The database auto-migrates on startup. No manual steps needed for updates.

If you hit a database error after updating:

```powershell
# Option A: Pull latest (auto-migration fixes it)
git pull origin main
# Then restart

# Option B: Reset database (loses all data)
Remove-Item data\arbitrage.db -ErrorAction SilentlyContinue
Remove-Item backend\arbitrage.db -ErrorAction SilentlyContinue
# Then restart
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `python` opens Microsoft Store | Disable app aliases: Settings > Apps > App execution aliases > turn off "python.exe" and "python3.exe" |
| `.\setup.ps1` is blocked by execution policy | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` then retry |
| `venv\Scripts\Activate.ps1` won't run | Same fix: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| Port already in use | Find and kill: `netstat -ano | findstr :8000` then `taskkill /PID <pid> /F` |
| `npm` not found | Restart your terminal after installing Node.js |
| Backend won't start | Check `python --version` (need 3.10+) |
| Frontend won't start | Check `node --version` (need 18+), run `cd frontend; npm install` |
| Missing `.env` | Run `copy .env.example .env` |
| `lsof` not found | Expected — `lsof` is a Unix command. Use `netstat` on Windows (see above) |
| Path too long errors | Enable long paths: `git config --system core.longpaths true` |
