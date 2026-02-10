# Homerun - Windows Run Script (TUI)
# Run: .\run.ps1

$ErrorActionPreference = "Stop"

# Check if setup was run
if (-not (Test-Path "backend\venv")) {
    Write-Host "Setup not complete. Running setup first..." -ForegroundColor Yellow
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
