# Homerun - Windows Run Script
# Run: .\run.ps1

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host " _   _  ___  __  __ _____ ____  _   _ _   _" -ForegroundColor Green
Write-Host "| | | |/ _ \|  \/  | ____|  _ \| | | | \ | |" -ForegroundColor Green
Write-Host "| |_| | | | | |\/| |  _| | |_) | | | |  \| |" -ForegroundColor Green
Write-Host "|  _  | |_| | |  | | |___|  _ <| |_| | |\  |" -ForegroundColor Green
Write-Host "|_| |_|\___/|_|  |_|_____|_| \_\\___/|_| \_|" -ForegroundColor Green
Write-Host ""
Write-Host "Autonomous Prediction Market Trading Platform" -ForegroundColor Cyan
Write-Host ""

# Check if setup was run
if (-not (Test-Path "backend\venv")) {
    Write-Host "Setup not complete. Running setup first..." -ForegroundColor Yellow
    & .\setup.ps1
}

# Kill any process using a given port
function Stop-PortProcess {
    param([int]$Port)
    $connections = netstat -ano | Select-String ":$Port\s" | Select-String "LISTENING"
    foreach ($conn in $connections) {
        $parts = $conn -split '\s+'
        $pid = $parts[-1]
        if ($pid -and $pid -ne "0") {
            Write-Host "Killing existing process on port $Port (PID: $pid)" -ForegroundColor Yellow
            try { Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue } catch {}
        }
    }
}

# Clean up ports
Stop-PortProcess -Port 8000
Stop-PortProcess -Port 3000

# Start backend
Write-Host "Starting backend..." -ForegroundColor Cyan
$backendJob = Start-Job -ScriptBlock {
    Set-Location $using:PWD\backend
    & .\venv\Scripts\Activate.ps1
    uvicorn main:app --host 0.0.0.0 --port 8000
}

# Wait for backend
Write-Host "Waiting for backend to start..."
Start-Sleep -Seconds 4

# Check backend health
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing -TimeoutSec 5 -ErrorAction SilentlyContinue
    Write-Host "Backend is healthy." -ForegroundColor Green
} catch {
    Write-Host "Warning: Backend may not have started correctly." -ForegroundColor Yellow
}

# Start frontend
Write-Host "Starting frontend..." -ForegroundColor Cyan
$frontendJob = Start-Job -ScriptBlock {
    Set-Location $using:PWD\frontend
    npm run dev
}

Write-Host ""
Write-Host "=========================================" -ForegroundColor Green
Write-Host "  HOMERUN is running!" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Dashboard: http://localhost:3000" -ForegroundColor Cyan
Write-Host "  API:       http://localhost:8000" -ForegroundColor Cyan
Write-Host "  API Docs:  http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
Write-Host ""

# Stream output and wait
try {
    while ($true) {
        # Print any output from backend/frontend jobs
        Receive-Job -Job $backendJob -ErrorAction SilentlyContinue | Write-Host
        Receive-Job -Job $frontendJob -ErrorAction SilentlyContinue | Write-Host

        # Check if jobs are still running
        if ($backendJob.State -eq "Failed") {
            Write-Host "Backend process exited unexpectedly." -ForegroundColor Red
            Receive-Job -Job $backendJob -ErrorAction SilentlyContinue | Write-Host
        }
        if ($frontendJob.State -eq "Failed") {
            Write-Host "Frontend process exited unexpectedly." -ForegroundColor Red
            Receive-Job -Job $frontendJob -ErrorAction SilentlyContinue | Write-Host
        }

        Start-Sleep -Seconds 2
    }
} finally {
    Write-Host ""
    Write-Host "Shutting down..." -ForegroundColor Yellow
    Stop-Job -Job $backendJob -ErrorAction SilentlyContinue
    Stop-Job -Job $frontendJob -ErrorAction SilentlyContinue
    Remove-Job -Job $backendJob -ErrorAction SilentlyContinue
    Remove-Job -Job $frontendJob -ErrorAction SilentlyContinue
    Stop-PortProcess -Port 8000
    Stop-PortProcess -Port 3000
    Write-Host "Stopped." -ForegroundColor Green
}
