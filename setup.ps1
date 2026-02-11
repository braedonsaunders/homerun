# Homerun - Windows Setup Script
# Run: .\setup.ps1

$ErrorActionPreference = "Stop"

Write-Host "=========================================" -ForegroundColor Green
Write-Host "  Homerun Setup (Windows)" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green
Write-Host ""

# Check Python
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Found $pythonVersion"
    $versionMatch = [regex]::Match($pythonVersion, '(\d+)\.(\d+)')
    $major = [int]$versionMatch.Groups[1].Value
    $minor = [int]$versionMatch.Groups[2].Value
    if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 10)) {
        Write-Host "Warning: Python 3.10+ recommended. You have $pythonVersion" -ForegroundColor Yellow
    }
} catch {
    Write-Host "Error: Python is required but not found in PATH." -ForegroundColor Red
    Write-Host "Download from https://www.python.org/downloads/" -ForegroundColor Yellow
    Write-Host "Make sure to check 'Add Python to PATH' during installation." -ForegroundColor Yellow
    exit 1
}

# Check Node.js
try {
    $nodeVersion = node --version 2>&1
    Write-Host "Found Node.js $nodeVersion"
} catch {
    Write-Host "Error: Node.js is required but not found in PATH." -ForegroundColor Red
    Write-Host "Download from https://nodejs.org/" -ForegroundColor Yellow
    exit 1
}

# Create .env if missing
if (-not (Test-Path ".env")) {
    Write-Host ""
    Write-Host "Creating .env file from template..."
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env - edit this file to configure settings"
}

# Setup backend
Write-Host ""
Write-Host "Setting up backend..." -ForegroundColor Cyan

Push-Location backend

if (-not (Test-Path "venv")) {
    Write-Host "Creating Python virtual environment..."
    python -m venv venv
}

Write-Host "Activating virtual environment..."
& .\venv\Scripts\Activate.ps1

Write-Host "Installing Python dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

# Try trading dependencies
if ($minor -ge 10) {
    Write-Host "Installing trading dependencies..."
    try {
        pip install --quiet -r requirements-trading.txt 2>$null
    } catch {
        Write-Host "  (trading deps skipped - optional)" -ForegroundColor Yellow
    }
} else {
    Write-Host ""
    Write-Host "Note: Python 3.10+ required for live trading." -ForegroundColor Yellow
    Write-Host "      Paper trading and scanning will work fine."
}

Pop-Location

# Setup frontend
Write-Host ""
Write-Host "Setting up frontend..." -ForegroundColor Cyan

Push-Location frontend

Write-Host "Installing Node.js dependencies..."
npm install --silent 2>$null
if ($LASTEXITCODE -ne 0) {
    npm install
}

Pop-Location

# Create data directory
if (-not (Test-Path "data")) {
    New-Item -ItemType Directory -Path "data" | Out-Null
}

# Write setup fingerprint so run.ps1 can detect drift and auto-rerun setup.
function Get-HashOrMissing {
    param([string]$Path)
    if (-not (Test-Path $Path)) { return "missing" }
    return (Get-FileHash -Path $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

$stamp = @{
    python_version = (python -c "import platform; print(platform.python_version())")
    requirements_sha256 = Get-HashOrMissing "backend\requirements.txt"
    requirements_trading_sha256 = Get-HashOrMissing "backend\requirements-trading.txt"
    package_json_sha256 = Get-HashOrMissing "frontend\package.json"
    package_lock_sha256 = Get-HashOrMissing "frontend\package-lock.json"
}

$stamp | ConvertTo-Json | Set-Content -Path ".setup-stamp.json" -Encoding UTF8
Write-Host "Wrote .setup-stamp.json"

Write-Host ""
Write-Host "=========================================" -ForegroundColor Green
Write-Host "  Setup Complete!" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green
Write-Host ""
Write-Host "To start the application, run:"
Write-Host "  .\run.ps1" -ForegroundColor Cyan
Write-Host ""
Write-Host "Or start services individually:"
Write-Host "  Backend:  cd backend; .\venv\Scripts\Activate.ps1; uvicorn main:app --reload" -ForegroundColor Gray
Write-Host "  Frontend: cd frontend; npm run dev" -ForegroundColor Gray
Write-Host ""
Write-Host "The app will be available at:"
Write-Host "  Frontend: http://localhost:3000" -ForegroundColor Cyan
Write-Host "  Backend:  http://localhost:8000" -ForegroundColor Cyan
Write-Host "  API Docs: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host ""
