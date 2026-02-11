# Homerun - Windows Run Script (TUI)
# Run: .\run.ps1

$ErrorActionPreference = "Stop"

function Test-NeedsSetup {
    if (-not (Test-Path "backend\venv")) { return $true }
    if (-not (Test-Path "frontend\node_modules")) { return $true }
    if (-not (Test-Path ".setup-stamp.json")) { return $true }

    try {
        $stamp = Get-Content ".setup-stamp.json" -Raw | ConvertFrom-Json
    } catch {
        return $true
    }

    function Get-HashOrMissing {
        param([string]$Path)
        if (-not (Test-Path $Path)) { return "missing" }
        return (Get-FileHash -Path $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    }

    $pythonVersion = (python -c "import platform; print(platform.python_version())")
    if ($stamp.python_version -ne $pythonVersion) { return $true }
    if ($stamp.requirements_sha256 -ne (Get-HashOrMissing "backend\requirements.txt")) { return $true }
    if ($stamp.requirements_trading_sha256 -ne (Get-HashOrMissing "backend\requirements-trading.txt")) { return $true }
    if ($stamp.package_json_sha256 -ne (Get-HashOrMissing "frontend\package.json")) { return $true }
    if ($stamp.package_lock_sha256 -ne (Get-HashOrMissing "frontend\package-lock.json")) { return $true }

    return $false
}

if (Test-NeedsSetup) {
    Write-Host "Setup missing or stale. Running setup..." -ForegroundColor Yellow
    & .\setup.ps1
}

# Activate venv
& backend\venv\Scripts\Activate.ps1

# Ensure TUI dependencies are installed
try {
    python -c "import textual" 2>$null
} catch {
    Write-Host "Installing TUI dependencies..." -ForegroundColor Cyan
    pip install -q textual rich
}

# Launch the TUI
python tui.py @args
