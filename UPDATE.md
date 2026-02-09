# Updating Homerun

This guide covers how to update Homerun to the latest version.

---

## Standard Update (Recommended)

```bash
# 1. Stop the running app (Ctrl+C if running)

# 2. Pull latest code
git pull origin main

# 3. Re-run setup to install any new dependencies
./setup.sh

# 4. Launch the app
./run.sh
```

The database will automatically migrate on startup -- any new columns are added to existing tables without losing data.

---

## If You Hit a Database Error

If you see an error like:

```
sqlite3.OperationalError: no such column: copy_trading_configs.copy_mode
```

This means your database was created with an older schema. Two options:

### Option A: Update the code (fixes it automatically)

Pull the latest version which includes auto-migration:

```bash
git pull origin main
./run.sh
```

The app now detects missing columns and adds them on startup.

### Option B: Reset the database (loses all data)

If you want a clean start:

```bash
rm -f data/arbitrage.db backend/arbitrage.db
./run.sh
```

This deletes the old database. A fresh one will be created on startup.

---

## Update on Windows

```powershell
# 1. Stop the running app (Ctrl+C if running)

# 2. Pull latest code
git pull origin main

# 3. Re-run setup
.\setup.ps1

# 4. Launch the app
.\run.ps1
```

Or manually:

```powershell
git pull origin main
cd backend
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
cd ..\frontend
npm install
```

Then start both services again. See [WINDOWS.md](WINDOWS.md) for full Windows instructions.

If you need to reset the database on Windows:

```powershell
Remove-Item data\arbitrage.db -ErrorAction SilentlyContinue
Remove-Item backend\arbitrage.db -ErrorAction SilentlyContinue
```

---

## Update with Docker

```bash
git pull origin main
docker-compose down
docker-compose up --build
```

---

## Update with Make

```bash
git pull origin main
make setup
make run
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `no such column` errors | Pull latest code (has auto-migration) or delete `data/arbitrage.db` |
| Backend won't start | Check `python3 --version` (need 3.10+) |
| Frontend won't start | Check `node --version` (need 18+), run `cd frontend && npm install` |
| Port already in use | Kill existing process: `lsof -ti:8000 \| xargs kill` |
| Missing `.env` | Run `cp .env.example .env` |
