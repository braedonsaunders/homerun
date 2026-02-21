# Homerun - Windows Setup Script
# Run: .\scripts\setup.ps1

param(
    [switch]$RedisOnly,
    [switch]$PostgresOnly
)

$ErrorActionPreference = "Stop"

# Navigate to project root (parent of scripts\)
Set-Location (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))

Write-Host "=========================================" -ForegroundColor Green
Write-Host "  Homerun Setup (Windows)" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green
Write-Host ""

function Find-RedisServer {
    $cmd = Get-Command redis-server -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }

    $wellKnown = "C:\Program Files\Redis\redis-server.exe"
    if (Test-Path $wellKnown) { return $wellKnown }

    return $null
}

function Find-PostgresBinDir {
    if ($env:POSTGRES_BIN_DIR) {
        $envBin = $env:POSTGRES_BIN_DIR
        if (
            (Test-Path (Join-Path $envBin "initdb.exe")) -and
            (Test-Path (Join-Path $envBin "pg_ctl.exe"))
        ) {
            return $envBin
        }
    }

    $initdb = Get-Command initdb -ErrorAction SilentlyContinue
    $pgctl = Get-Command pg_ctl -ErrorAction SilentlyContinue
    if ($initdb -and $pgctl) {
        return (Split-Path -Parent $initdb.Source)
    }

    $roots = @(
        "C:\Program Files\PostgreSQL",
        "C:\Program Files (x86)\PostgreSQL"
    )
    foreach ($root in $roots) {
        if (-not (Test-Path $root)) {
            continue
        }
        $versions = Get-ChildItem -Path $root -Directory -ErrorAction SilentlyContinue | Sort-Object Name -Descending
        foreach ($versionDir in $versions) {
            $bin = Join-Path $versionDir.FullName "bin"
            if (
                (Test-Path (Join-Path $bin "initdb.exe")) -and
                (Test-Path (Join-Path $bin "pg_ctl.exe"))
            ) {
                return $bin
            }
        }
    }

    return $null
}

function Test-RedisRuntimeAvailable {
    if (Get-Command docker -ErrorAction SilentlyContinue) { return $true }
    if (Find-RedisServer) { return $true }
    try {
        $svc = Get-Service -Name "Memurai" -ErrorAction SilentlyContinue
        if ($svc) { return $true }
    } catch {
    }
    return $false
}

function Ensure-RedisRuntime {
    if (Test-RedisRuntimeAvailable) {
        Write-Host "Redis runtime prerequisite found (docker, redis-server, or Memurai)." -ForegroundColor Green
        return $true
    }

    Write-Host "Redis runtime prerequisite missing. Attempting to install..." -ForegroundColor Cyan

    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if ($winget) {
        $wingetIds = @(
            "Redis.Redis",
            "Memurai.MemuraiDeveloper"
        )
        foreach ($id in $wingetIds) {
            try {
                winget install --id $id --exact --silent --accept-source-agreements --accept-package-agreements *> $null
                if (Test-RedisRuntimeAvailable) {
                    Write-Host "Redis runtime installed via winget ($id)." -ForegroundColor Green
                    return $true
                }
            } catch {
            }
        }
    }

    $choco = Get-Command choco -ErrorAction SilentlyContinue
    if ($choco) {
        try {
            choco install redis-64 -y *> $null
            if (Test-RedisRuntimeAvailable) {
                Write-Host "Redis runtime installed via Chocolatey." -ForegroundColor Green
                return $true
            }
        } catch {
        }
    }

    Write-Host "Failed to auto-install Redis runtime prerequisites." -ForegroundColor Red
    Write-Host "Install Docker Desktop, redis-server, or Memurai, then rerun setup." -ForegroundColor Yellow
    return $false
}

function Test-PostgresRuntimeAvailable {
    if (Get-Command docker -ErrorAction SilentlyContinue) { return $true }
    return [bool](Find-PostgresBinDir)
}

function Ensure-PostgresRuntime {
    if (Test-PostgresRuntimeAvailable) {
        Write-Host "Postgres runtime prerequisite found (docker or initdb+pg_ctl)." -ForegroundColor Green
        return $true
    }

    Write-Host "Postgres runtime prerequisite missing. Attempting to install..." -ForegroundColor Cyan

    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if ($winget) {
        $wingetIds = @(
            "PostgreSQL.PostgreSQL",
            "PostgreSQL.PostgreSQL.16"
        )
        foreach ($id in $wingetIds) {
            try {
                winget install --id $id --exact --silent --accept-source-agreements --accept-package-agreements *> $null
                if (Test-PostgresRuntimeAvailable) {
                    Write-Host "Postgres runtime installed via winget ($id)." -ForegroundColor Green
                    return $true
                }
            } catch {
            }
        }
    }

    $choco = Get-Command choco -ErrorAction SilentlyContinue
    if ($choco) {
        try {
            choco install postgresql -y *> $null
            if (Test-PostgresRuntimeAvailable) {
                Write-Host "Postgres runtime installed via Chocolatey." -ForegroundColor Green
                return $true
            }
        } catch {
        }
    }

    Write-Host "Failed to auto-install Postgres runtime prerequisites." -ForegroundColor Red
    Write-Host "Install Docker Desktop or PostgreSQL tools (initdb + pg_ctl), then rerun setup." -ForegroundColor Yellow
    return $false
}

if ($RedisOnly -and $PostgresOnly) {
    if (-not (Ensure-RedisRuntime)) { exit 1 }
    if (-not (Ensure-PostgresRuntime)) { exit 1 }
    exit 0
}

if ($RedisOnly) {
    if (-not (Ensure-RedisRuntime)) {
        exit 1
    }
    exit 0
}

if ($PostgresOnly) {
    if (-not (Ensure-PostgresRuntime)) {
        exit 1
    }
    exit 0
}

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

Write-Host ""
Write-Host "Setting up launcher tooling..." -ForegroundColor Cyan
$originalCxxFlags = $env:CXXFLAGS
$env:CXXFLAGS = "$($env:CXXFLAGS) -std=c++20".Trim()
npm --prefix scripts/tooling install --silent 2>$null
if ($LASTEXITCODE -ne 0) {
    npm --prefix scripts/tooling install
}
$env:CXXFLAGS = $originalCxxFlags

Write-Host "Verifying PowerShell launcher syntax..." -ForegroundColor Cyan
node .\scripts\tooling\check_powershell_syntax.mjs .\scripts\run.ps1 .\scripts\setup.ps1
if ($LASTEXITCODE -ne 0) {
    Write-Host "PowerShell syntax verification failed." -ForegroundColor Red
    exit 1
}

# Create data directory
if (-not (Test-Path "data")) {
    New-Item -ItemType Directory -Path "data" | Out-Null
}

Write-Host ""
Write-Host "Ensuring Redis runtime prerequisites..." -ForegroundColor Cyan
if (-not (Ensure-RedisRuntime)) {
    exit 1
}

Write-Host "Ensuring Postgres runtime prerequisites..." -ForegroundColor Cyan
if (-not (Ensure-PostgresRuntime)) {
    exit 1
}

# Write setup fingerprint so run.ps1 can detect drift and auto-rerun setup.
function Get-HashOrMissing {
    param([string]$Path)
    if (-not (Test-Path $Path)) { return "missing" }
    return (Get-FileHash -Path $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

$stamp = @{
    python_version = (& .\backend\venv\Scripts\python.exe -c "import platform; print(platform.python_version())")
    requirements_sha256 = Get-HashOrMissing "backend\requirements.txt"
    requirements_trading_sha256 = Get-HashOrMissing "backend\requirements-trading.txt"
    package_json_sha256 = Get-HashOrMissing "frontend\package.json"
    package_lock_sha256 = Get-HashOrMissing "frontend\package-lock.json"
    launcher_tools_package_json_sha256 = Get-HashOrMissing "scripts\tooling\package.json"
    launcher_tools_package_lock_sha256 = Get-HashOrMissing "scripts\tooling\package-lock.json"
}

$stamp | ConvertTo-Json | Set-Content -Path ".setup-stamp.json" -Encoding UTF8
Write-Host "Wrote .setup-stamp.json"

Write-Host ""
Write-Host "=========================================" -ForegroundColor Green
Write-Host "  Setup Complete!" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green
Write-Host ""
Write-Host "To start the application, run:"
Write-Host "  .\scripts\run.ps1" -ForegroundColor Cyan
Write-Host ""
Write-Host "Or run runtime validation only:"
Write-Host "  .\scripts\run.ps1 --services-smoke-test" -ForegroundColor Cyan
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
